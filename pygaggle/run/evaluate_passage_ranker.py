from typing import Optional, List, Tuple
from pathlib import Path
import logging

from pydantic import BaseModel, validator
from transformers import (AutoModel,
                          AutoTokenizer,
                          BertForSequenceClassification)
import torch

from .args import ArgumentParserBuilder, opt
from pygaggle.rerank.base import Reranker
from pygaggle.rerank.bm25 import Bm25Reranker
from pygaggle.rerank.transformer import (
    UnsupervisedTransformerReranker,
    MonoT5,
    DuoT5,
    MonoBERT
)
from pygaggle.rerank.random import RandomReranker
from pygaggle.rerank.similarity import CosineSimilarityMatrixProvider
from pygaggle.model import (SimpleBatchTokenizer,
                            RerankerEvaluator,
                            DuoRerankerEvaluator,
                            metric_names,
                            MsMarcoWriter)
from pygaggle.data import MsMarcoDataset
from pygaggle.settings import MsMarcoSettings


SETTINGS = MsMarcoSettings()
METHOD_CHOICES = ('transformer', 'bm25', 't5', 'seq_class_transformer',
                  'random', 'duo_t5')


class PassageRankingEvaluationOptions(BaseModel):
    task: str
    dataset: Path
    index_dir: Path
    method: str
    model: str
    duo_model: str
    mono_hits: int
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

    @validator('model')
    def model_sane(cls, v: str, values, **kwargs):
        method = values['method']
        if method == 'transformer' and v is None:
            raise ValueError('transformer name or path must be specified')
        return v

    @validator('tokenizer_name')
    def tokenizer_sane(cls, v: str, values, **kwargs):
        if v is None:
            return values['model']
        return v


def construct_t5(options: PassageRankingEvaluationOptions) -> Reranker:
    model = MonoT5.get_model(options.model,
                             from_tf=options.from_tf,
                             device=options.device)
    tokenizer = MonoT5.get_tokenizer(options.model_type, batch_size=options.batch_size)
    return MonoT5(model, tokenizer)


def construct_duo_t5(options: PassageRankingEvaluationOptions) -> Tuple[Reranker, Reranker]:
    mono_reranker = construct_t5(options)
    model = DuoT5.get_model(options.duo_model,
                            from_tf=options.from_tf,
                            device=options.device)
    tokenizer = DuoT5.get_tokenizer(options.model_type, batch_size=options.batch_size)
    return mono_reranker, DuoT5(model, tokenizer)


def construct_transformer(options:
                          PassageRankingEvaluationOptions) -> Reranker:
    device = torch.device(options.device)
    model = AutoModel.from_pretrained(options.model,
                                      from_tf=options.from_tf).to(device).eval()
    tokenizer = SimpleBatchTokenizer(AutoTokenizer.from_pretrained(
        options.tokenizer_name, use_fast=False),
        options.batch_size)
    provider = CosineSimilarityMatrixProvider()
    return UnsupervisedTransformerReranker(model, tokenizer, provider)


def construct_seq_class_transformer(options: PassageRankingEvaluationOptions
                                    ) -> Reranker:
    try:
        model = MonoBERT.get_model(
            options.model, from_tf=options.from_tf, device=options.device)
    except AttributeError:
        # Hotfix for BioBERT MS MARCO. Refactor.
        BertForSequenceClassification.bias = torch.nn.Parameter(
            torch.zeros(2))
        BertForSequenceClassification.weight = torch.nn.Parameter(
            torch.zeros(2, 768))
        model = BertForSequenceClassification.from_pretrained(
            options.model, from_tf=options.from_tf)
        model.classifier.weight = BertForSequenceClassification.weight
        model.classifier.bias = BertForSequenceClassification.bias
        device = torch.device(options.device)
        model = model.to(device).eval()
    tokenizer = MonoBERT.get_tokenizer(options.tokenizer_name)
    return MonoBERT(model, tokenizer)


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
                 opt('--model',
                     required=True,
                     type=str,
                     help='Path to pre-trained model or huggingface model name'),
                 opt('--duo_model',
                     type=str,
                     default='',
                     help='Path to pre-trained model or huggingface model name'),
                 opt('--mono_hits',
                     type=int,
                     default=50,
                     help='Top k candidates from mono for duo reranking'),
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
                 opt('--model-type', type=str),
                 opt('--tokenizer-name', type=str))
    args = apb.parser.parse_args()
    options = PassageRankingEvaluationOptions(**vars(args))
    logging.info("Preprocessing Queries & Passages:")
    ds = MsMarcoDataset.from_folder(str(options.dataset), split=options.split,
                                    is_duo=options.is_duo)
    examples = ds.to_relevance_examples(str(options.index_dir),
                                        is_duo=options.is_duo)
    logging.info("Loading Ranker & Tokenizer:")
    construct_map = dict(transformer=construct_transformer,
                         bm25=construct_bm25,
                         t5=construct_t5,
                         duo_t5=construct_duo_t5,
                         seq_class_transformer=construct_seq_class_transformer,
                         random=lambda _: RandomReranker())
    reranker = construct_map[options.method](options)
    writer = MsMarcoWriter(args.output_file, args.overwrite_output)
    if options.method == 'duo_t5':
        evaluator = DuoRerankerEvaluator(mono_reranker=reranker[0],
                                         duo_reranker=reranker[1],
                                         metric_names=options.metrics,
                                         mono_hits=options.mono_hits,
                                         writer=writer)
    else:
        evaluator = RerankerEvaluator(reranker, options.metrics, writer=writer)
    width = max(map(len, args.metrics)) + 1
    logging.info("Reranking:")
    for metric in evaluator.evaluate(examples):
        logging.info(f'{metric.name:<{width}}{metric.value:.5}')


if __name__ == '__main__':
    main()
