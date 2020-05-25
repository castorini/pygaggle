from typing import Optional, List
from pathlib import Path
import logging

from pydantic import BaseModel, validator
from transformers import (AutoModel,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          BertForQuestionAnswering,
                          BertForSequenceClassification)
import torch

from .args import ArgumentParserBuilder, opt
from pygaggle.rerank.base import Reranker
from pygaggle.rerank.bm25 import Bm25Reranker
from pygaggle.rerank.transformer import (
    QuestionAnsweringTransformerReranker,
    SequenceClassificationTransformerReranker,
    T5Reranker,
    UnsupervisedTransformerReranker
    )
from pygaggle.rerank.random import RandomReranker
from pygaggle.rerank.similarity import CosineSimilarityMatrixProvider
from pygaggle.model import (CachedT5ModelLoader,
                            RerankerEvaluator,
                            SimpleBatchTokenizer,
                            T5BatchTokenizer,
                            metric_names)
from pygaggle.data import LitReviewDataset
from pygaggle.settings import Cord19Settings


SETTINGS = Cord19Settings()
METHOD_CHOICES = ('transformer', 'bm25', 't5', 'seq_class_transformer',
                  'qa_transformer', 'random')


class KaggleEvaluationOptions(BaseModel):
    dataset: Path
    index_dir: Path
    method: str
    batch_size: int
    device: str
    split: str
    metrics: List[str]
    model_name: Optional[str]
    tokenizer_name: Optional[str]

    @validator('dataset')
    def dataset_exists(cls, v: Path):
        assert v.exists(), 'dataset must exist'
        return v

    @validator('model_name')
    def model_name_sane(cls, v: Optional[str], values, **kwargs):
        method = values['method']
        if method == 'transformer' and v is None:
            raise ValueError('transformer name must be specified')
        elif method == 't5':
            return SETTINGS.t5_model_type
        if v == 'biobert':
            return 'monologg/biobert_v1.1_pubmed'
        return v

    @validator('tokenizer_name')
    def tokenizer_sane(cls, v: str, values, **kwargs):
        if v is None:
            return values['model_name']
        return v


def construct_t5(options: KaggleEvaluationOptions) -> Reranker:
    loader = CachedT5ModelLoader(SETTINGS.t5_model_dir,
                                 SETTINGS.cache_dir,
                                 'ranker',
                                 SETTINGS.t5_model_type,
                                 SETTINGS.flush_cache)
    device = torch.device(options.device)
    model = loader.load().to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
                    options.model_name)
    tokenizer = T5BatchTokenizer(tokenizer, options.batch_size)
    return T5Reranker(model, tokenizer)


def construct_transformer(options: KaggleEvaluationOptions) -> Reranker:
    device = torch.device(options.device)
    try:
        model = AutoModel.from_pretrained(options.model_name).to(device).eval()
    except OSError:
        model = AutoModel.from_pretrained(options.model_name,
                                          from_tf=True).to(device).eval()
    tokenizer = SimpleBatchTokenizer(
                    AutoTokenizer.from_pretrained(
                        options.tokenizer_name),
                    options.batch_size)
    provider = CosineSimilarityMatrixProvider()
    return UnsupervisedTransformerReranker(model, tokenizer, provider)


def construct_seq_class_transformer(options:
                                    KaggleEvaluationOptions) -> Reranker:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
                    options.model_name)
    except OSError:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                        options.model_name,
                        from_tf=True)
        except AttributeError:
            # Hotfix for BioBERT MS MARCO. Refactor.
            BertForSequenceClassification.bias = torch.nn.Parameter(
                                                    torch.zeros(2))
            BertForSequenceClassification.weight = torch.nn.Parameter(
                                                    torch.zeros(2, 768))
            model = BertForSequenceClassification.from_pretrained(
                        options.model_name, from_tf=True)
            model.classifier.weight = BertForSequenceClassification.weight
            model.classifier.bias = BertForSequenceClassification.bias
    device = torch.device(options.device)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
                    options.tokenizer_name)
    return SequenceClassificationTransformerReranker(model, tokenizer)


def construct_qa_transformer(options: KaggleEvaluationOptions) -> Reranker:
    # We load a sequence classification model first -- again, as a workaround.
    # Refactor
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
                    options.model_name)
    except OSError:
        model = AutoModelForSequenceClassification.from_pretrained(
                    options.model_name, from_tf=True)
    fixed_model = BertForQuestionAnswering(model.config)
    fixed_model.qa_outputs = model.classifier
    fixed_model.bert = model.bert
    device = torch.device(options.device)
    model = fixed_model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
                    options.tokenizer_name)
    return QuestionAnsweringTransformerReranker(model, tokenizer)


def construct_bm25(options: KaggleEvaluationOptions) -> Reranker:
    return Bm25Reranker(index_path=str(options.index_dir))


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(opt('--dataset', type=Path, required=True),
                 opt('--index-dir', type=Path, required=True),
                 opt('--method',
                     required=True,
                     type=str,
                     choices=METHOD_CHOICES),
                 opt('--model-name', type=str),
                 opt('--split', type=str, default='nq', choices=('nq', 'kq')),
                 opt('--batch-size', '-bsz', type=int, default=96),
                 opt('--device', type=str, default='cuda:0'),
                 opt('--tokenizer-name', type=str),
                 opt('--metrics',
                     type=str,
                     nargs='+',
                     default=metric_names(),
                     choices=metric_names()))
    args = apb.parser.parse_args()
    options = KaggleEvaluationOptions(**vars(args))
    ds = LitReviewDataset.from_file(str(options.dataset))
    examples = ds.to_senticized_dataset(str(options.index_dir),
                                        split=options.split)
    construct_map = dict(transformer=construct_transformer,
                         bm25=construct_bm25,
                         t5=construct_t5,
                         seq_class_transformer=construct_seq_class_transformer,
                         qa_transformer=construct_qa_transformer,
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
