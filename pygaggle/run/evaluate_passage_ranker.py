from typing import Optional, List
from pathlib import Path
import logging

from pydantic import BaseModel, validator
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
import torch

from .args import ArgumentParserBuilder, opt
from pygaggle.rerank.base import Reranker
from pygaggle.rerank.bm25 import Bm25Reranker
from pygaggle.rerank.transformer import UnsupervisedTransformerReranker, T5Reranker, \
    SequenceClassificationTransformerReranker
from pygaggle.rerank.random import RandomReranker
from pygaggle.rerank.similarity import CosineSimilarityMatrixProvider
from pygaggle.model import SimpleBatchTokenizer, CachedT5ModelLoader, T5BatchTokenizer, RerankerEvaluator, metric_names
from pygaggle.data import MsMarcoDataset
from pygaggle.settings import MsMarcoSettings


SETTINGS = MsMarcoSettings()
METHOD_CHOICES = ('transformer', 'bm25', 't5', 'seq_class_transformer', 'random')


class PassageRankingEvaluationOptions(BaseModel):
    dataset: str
    data_dir: Path
    method: str
    batch_size: int
    device: str
    split: str
    do_lower_case: bool
    is_duo: bool
    metrics: List[str]
    model_name: Optional[str]
    tokenizer_name: Optional[str]

    @validator('dataset')
    def dataset_exists(cls, v: str):
        assert v in ['msmarco', 'treccar']

    @validator('data_dir')
    def datadir_exists(cls, v: str):
        assert v.exists(), 'data directory must exist'
        return v

    #TODO verify
    @validator('model_name')
    def model_name_sane(cls, v: Optional[str], values, **kwargs):
        method = values['method']
        if method == 'transformer' and v is None:
            raise ValueError('transformer name must be specified')
        elif method == 't5':
            return SETTINGS.t5_model_type
        if v == 'biobert':
            return 'monologg/biobert_v1.1_pubmed'
        if v == 'bert' and not is_duo:
            return SETTINGS.monobert_dir
        return v

    #TODO verify
    @validator('tokenizer_name')
    def tokenizer_sane(cls, v: str, values, **kwargs):
        if v is None:
            return values['model_name']
        return v


def construct_t5(options: PassageRankingEvaluationOptions) -> Reranker:
    loader = CachedT5ModelLoader(SETTINGS.t5_model_dir,
                                 SETTINGS.cache_dir,
                                 'ranker',
                                 SETTINGS.t5_model_type,
                                 SETTINGS.flush_cache)
    device = torch.device(options.device)
    model = loader.load().to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(options.model_name, do_lower_case=options.do_lower_case)
    tokenizer = T5BatchTokenizer(tokenizer, options.batch_size)
    return T5Reranker(model, tokenizer)

#TODO needed?
def construct_transformer(options: PassageRankingEvaluationOptions) -> Reranker:
    device = torch.device(options.device)
    try:
        model = AutoModel.from_pretrained(options.model_name).to(device).eval()
    except OSError:
        model = AutoModel.from_pretrained(options.model_name, from_tf=True).to(device).eval()
    tokenizer = SimpleBatchTokenizer(AutoTokenizer.from_pretrained(options.tokenizer_name,
                                                                   do_lower_case=options.do_lower_case),
                                     options.batch_size)
    provider = CosineSimilarityMatrixProvider() 
    return UnsupervisedTransformerReranker(model, tokenizer, provider)


def construct_seq_class_transformer(options: PassageRankingEvaluationOptions) -> Reranker:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(options.model_name)
    except OSError:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(options.model_name, from_tf=True)
        except AttributeError:
            # Hotfix for BioBERT MS MARCO. Refactor.
            BertForSequenceClassification.bias = torch.nn.Parameter(torch.zeros(2))
            BertForSequenceClassification.weight = torch.nn.Parameter(torch.zeros(2, 768))
            model = BertForSequenceClassification.from_pretrained(options.model_name, from_tf=True)
            model.classifier.weight = BertForSequenceClassification.weight
            model.classifier.bias = BertForSequenceClassification.bias
    device = torch.device(options.device)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(options.tokenizer_name, do_lower_case=options.do_lower_case)
    return SequenceClassificationTransformerReranker(model, tokenizer)


def construct_bm25(_: PassageRankingEvaluationOptions) -> Reranker:
    return Bm25Reranker(index_path=SETTINGS.msmarco_index_path)


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(opt('--dataset', type=str, default='msmarco'),
                 opt('--data-dir', type=Path, default='/content/data/msmarco'),
                 opt('--model-dir', type=Path, default='/content/models/msmarco'),
                 opt('--method', required=True, type=str, choices=METHOD_CHOICES),
                 opt('--model-name', type=str),
                 opt('--split', type=str, default='dev', choices=('dev', 'eval')),
                 opt('--batch-size', '-bsz', type=int, default=96),
                 opt('--device', type=str, default='cuda:0'),
                 opt('--tokenizer-name', type=str),
                 opt('--do-lower-case', action='store_true'),
                 opt('--is-duo', action='store_true'),
                 opt('--metrics', type=str, nargs='+', default=metric_names(), choices=metric_names()))
    args = apb.parser.parse_args()
    options = PassageRankingEvaluationOptions(**vars(args))
    ds = MsMarcoDataset.from_folder(str(options.data_dir), split=options.split, is_duo=options.is_duo)
    examples = ds.to_relevance_examples(SETTINGS.msmarco_index_path, is_duo=options.is_duo)
    construct_map = dict(transformer=construct_transformer,
                         bm25=construct_bm25,
                         t5=construct_t5,
                         seq_class_transformer=construct_seq_class_transformer,
                         random=lambda _: RandomReranker())
    reranker = construct_map[options.method](options)
    evaluator = RerankerEvaluator(reranker, options.metrics)
    width = max(map(len, args.metrics)) + 1
    stdout = []
    for metric in evaluator.evaluate(examples):
        logging.info(f'{metric.name:<{width}}{metric.value:.5}')
        stdout.append(f'{metric.name}\t{metric.value}')
    print('\n'.join(stdout))


if __name__ == '__main__':
    main()
