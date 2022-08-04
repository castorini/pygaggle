from typing import Optional, List
from pathlib import Path
import logging

from pydantic import BaseModel, validator
from transformers import (AutoModel,
                          AutoTokenizer)
import torch

from .args import ArgumentParserBuilder, opt
from pygaggle.rerank.base import Reranker
from pygaggle.rerank.bm25 import Bm25Reranker
from pygaggle.rerank.transformer import (
    UnsupervisedTransformerReranker,
    MonoT5,
    MonoBERT
)
from pygaggle.rerank.random import RandomReranker
from pygaggle.rerank.similarity import CosineSimilarityMatrixProvider
from pygaggle.model import (SimpleBatchTokenizer,
                            RerankerEvaluator,
                            metric_names,
                            MsMarcoWriter)
from pygaggle.data import MsMarcoDataset
from pygaggle.settings import MsMarcoSettings


SETTINGS = MsMarcoSettings()
METHOD_CHOICES = ('transformer', 'bm25', 't5', 'seq_class_transformer',
                  'random')


class DocumentRankingEvaluationOptions(BaseModel):
    task: str
    dataset: Path
    index_dir: Path
    method: str
    model: str
    split: str
    batch_size: int
    seg_size: int
    seg_stride: int
    aggregate_method: str
    device: str
    is_duo: bool
    from_tf: bool
    metrics: List[str]
    model_type: Optional[str]
    tokenizer_name: Optional[str]

    @validator('task')
    def task_exists(cls, v: str):
        assert v in ['msmarco']

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


def construct_t5(options: DocumentRankingEvaluationOptions) -> Reranker:
    model = MonoT5.get_model(options.model,
                             from_tf=options.from_tf,
                             device=options.device)
    tokenizer = MonoT5.get_tokenizer(options.model_type, batch_size=options.batch_size)
    return MonoT5(model = model, tokenizer = tokenizer)


def construct_transformer(options:
                          DocumentRankingEvaluationOptions) -> Reranker:
    device = torch.device(options.device)
    model = AutoModel.from_pretrained(options.model,
                                      from_tf=options.from_tf).to(device).eval()
    tokenizer = SimpleBatchTokenizer(AutoTokenizer.from_pretrained(
        options.tokenizer_name, use_fast=False,),
        options.batch_size)
    provider = CosineSimilarityMatrixProvider()
    return UnsupervisedTransformerReranker(model, tokenizer, provider)


def construct_seq_class_transformer(options: DocumentRankingEvaluationOptions
                                    ) -> Reranker:
    model = MonoBERT.get_model(options.model, from_tf=options.from_tf, device=options.device)
    tokenizer = MonoBERT.get_tokenizer(options.tokenizer_name)
    return MonoBERT(model, tokenizer)


def construct_bm25(options: DocumentRankingEvaluationOptions) -> Reranker:
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
                 opt('--tokenizer-name', type=str),
                 opt('--seg-size', type=int, default=10),
                 opt('--seg-stride', type=int, default=5),
                 opt('--aggregate-method', type=str, default="max"))
    args = apb.parser.parse_args()
    options = DocumentRankingEvaluationOptions(**vars(args))
    logging.info("Preprocessing Queries & Docs:")
    ds = MsMarcoDataset.from_folder(str(options.dataset), split=options.split,
                                    is_duo=options.is_duo)
    examples = ds.to_relevance_examples(str(options.index_dir),
                                        is_duo=options.is_duo)
    logging.info("Loading Ranker & Tokenizer:")
    construct_map = dict(transformer=construct_transformer,
                         bm25=construct_bm25,
                         t5=construct_t5,
                         seq_class_transformer=construct_seq_class_transformer,
                         random=lambda _: RandomReranker())
    reranker = construct_map[options.method](options)
    writer = MsMarcoWriter(args.output_file, args.overwrite_output)
    evaluator = RerankerEvaluator(reranker, options.metrics, writer=writer)
    width = max(map(len, args.metrics)) + 1
    logging.info("Reranking:")
    for metric in evaluator.evaluate_by_segments(examples,
                                                 options.seg_size,
                                                 options.seg_stride,
                                                 options.aggregate_method):
        logging.info(f'{metric.name:<{width}}{metric.value:.5}')


if __name__ == '__main__':
    main()
