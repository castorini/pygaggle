from typing import Optional, List
from pathlib import Path
import logging

from pydantic import BaseModel, validator
from transformers import (AutoModel,
                          AutoTokenizer,
                          AutoModelForSequenceClassification,
                          BertForSequenceClassification,
                          T5ForConditionalGeneration)
import torch

from .args import ArgumentParserBuilder, opt
from pygaggle.rerank.base import Reranker
from pygaggle.rerank.bm25 import Bm25Reranker
from pygaggle.rerank.transformer import (
    UnsupervisedTransformerReranker,
    T5Reranker,
    SequenceClassificationTransformerReranker
    )
from pygaggle.rerank.random import RandomReranker
from pygaggle.rerank.similarity import CosineSimilarityMatrixProvider
from pygaggle.model import (SimpleBatchTokenizer,
                            T5BatchTokenizer,
                            RerankerEvaluator,
                            metric_names,
                            MsMarcoWriter)
from pygaggle.data import MsMarcoDataset
from pygaggle.settings import MsMarcoSettings


SETTINGS = MsMarcoSettings()
METHOD_CHOICES = ('transformer', 'bm25', 't5', 'seq_class_transformer',
                  'random')


class PassageRankingEvaluationOptions(BaseModel):
    task: str
    dataset: Path
    index_dir: Path
    method: str
    model_name_or_path: str
    split: str
    batch_size: int
    device: str
    is_duo: bool
    from_tf: bool
    metrics: List[str]
    model_type: Optional[str]
    tokenizer_name: Optional[str]

    @validator('task')
    def task_exists(cls, v: str):
        assert v in ['msmarco', 'treccar']

    @validator('dataset')
    def dataset_exists(cls, v: Path):
        assert v.exists(), 'data directory must exist'
        return v

    @validator('index_dir')
    def index_dir_exists(cls, v: Path):
        assert v.exists(), 'index directory must exist'
        return v

    @validator('model_name_or_path')
    def model_name_sane(cls, v: Optional[str], values, **kwargs):
        method = values['method']
        if method == 'transformer' and v is None:
            raise ValueError('transformer name or path must be specified')
        return v

    @validator('tokenizer_name')
    def tokenizer_sane(cls, v: str, values, **kwargs):
        if v is None:
            return values['model_name_or_path']
        return v


def construct_t5(options: PassageRankingEvaluationOptions) -> Reranker:
    device = torch.device(options.device)
    model = T5ForConditionalGeneration.from_pretrained(options.model_name_or_path,
                                                       from_tf=options.from_tf).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(options.model_type)
    tokenizer = T5BatchTokenizer(tokenizer, options.batch_size)
    return T5Reranker(model, tokenizer)


def construct_transformer(options:
                          PassageRankingEvaluationOptions) -> Reranker:
    device = torch.device(options.device)
    model = AutoModel.from_pretrained(options.model_name_or_path,
                                      from_tf=options.from_tf).to(device).eval()
    tokenizer = SimpleBatchTokenizer(AutoTokenizer.from_pretrained(
                                        options.tokenizer_name),
                                     options.batch_size)
    provider = CosineSimilarityMatrixProvider()
    return UnsupervisedTransformerReranker(model, tokenizer, provider)


def construct_seq_class_transformer(options: PassageRankingEvaluationOptions
                                    ) -> Reranker:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            options.model_name_or_path, from_tf=options.from_tf)
    except AttributeError:
        # Hotfix for BioBERT MS MARCO. Refactor.
        BertForSequenceClassification.bias = torch.nn.Parameter(
                                                torch.zeros(2))
        BertForSequenceClassification.weight = torch.nn.Parameter(
                                                torch.zeros(2, 768))
        model = BertForSequenceClassification.from_pretrained(
                    options.model_name_or_path, from_tf=options.from_tf)
        model.classifier.weight = BertForSequenceClassification.weight
        model.classifier.bias = BertForSequenceClassification.bias
    device = torch.device(options.device)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(options.tokenizer_name)
    return SequenceClassificationTransformerReranker(model, tokenizer)


def construct_bm25(options: PassageRankingEvaluationOptions) -> Reranker:
    return Bm25Reranker(index_path=str(options.index_dir))


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(opt('--task',
                     type=str,
                     default='msmarco'),
                 opt('--dataset', type=Path, required=True),
                 opt('--index-dir', type=Path, required=True),
                 opt('--method',
                     required=True,
                     type=str,
                     choices=METHOD_CHOICES),
                 opt('--model-name-or-path', type=str),
                 opt('--output-file', type=Path, default='.'),
                 opt('--overwrite-output', action='store_true'),
                 opt('--split',
                     type=str,
                     default='dev',
                     choices=('dev', 'eval')),
                 opt('--batch-size', '-bsz', type=int, default=96),
                 opt('--device', type=str, default='cuda:0'),
                 opt('--is-duo', action='store_true'),
                 opt('--from-tf', action='store_true'),
                 opt('--metrics',
                     type=str,
                     nargs='+',
                     default=metric_names(),
                     choices=metric_names()),
                 opt('--model-type', type=str, default='bert-base'),
                 opt('--tokenizer-name', type=str))
    args = apb.parser.parse_args()
    options = PassageRankingEvaluationOptions(**vars(args))
    ds = MsMarcoDataset.from_folder(str(options.dataset), split=options.split,
                                    is_duo=options.is_duo)
    examples = ds.to_relevance_examples(str(options.index_dir),
                                        is_duo=options.is_duo)
    construct_map = dict(transformer=construct_transformer,
                         bm25=construct_bm25,
                         t5=construct_t5,
                         seq_class_transformer=construct_seq_class_transformer,
                         random=lambda _: RandomReranker())
    reranker = construct_map[options.method](options)
    writer = MsMarcoWriter(args.output_file, args.overwrite_output)
    evaluator = RerankerEvaluator(reranker, options.metrics, writer=writer)
    width = max(map(len, args.metrics)) + 1
    stdout = []
    for metric in evaluator.evaluate(examples):
        logging.info(f'{metric.name:<{width}}{metric.value:.5}')
        stdout.append(f'{metric.name}\t{metric.value}')
    print('\n'.join(stdout))


if __name__ == '__main__':
    main()
