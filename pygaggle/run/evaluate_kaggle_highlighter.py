from typing import Optional, List
from pathlib import Path
import logging

from pydantic import BaseModel, validator
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering,\
    BertForQuestionAnswering
import torch

from .args import ArgumentParserBuilder, opt
from pygaggle.rerank import UnsupervisedTransformerReranker, InnerProductMatrixProvider, Reranker, T5Reranker, \
    Bm25Reranker, SequenceClassificationTransformerReranker, QuestionAnsweringTransformerReranker
from pygaggle.model import SimpleBatchTokenizer, CachedT5ModelLoader, T5BatchTokenizer, RerankerEvaluator, metric_names
from pygaggle.data import LitReviewDataset
from pygaggle.settings import Settings


SETTINGS = Settings()
METHOD_CHOICES = ('transformer', 'bm25', 't5', 'seq_class_transformer', 'qa_transformer')


class KaggleEvaluationOptions(BaseModel):
    dataset: Path
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
    tokenizer = AutoTokenizer.from_pretrained(options.model_name)
    tokenizer = T5BatchTokenizer(tokenizer, options.batch_size)
    return T5Reranker(model, tokenizer)


def construct_transformer(options: KaggleEvaluationOptions) -> Reranker:
    device = torch.device(options.device)
    try:
        model = AutoModel.from_pretrained(options.model_name).to(device).eval()
    except OSError:
        model = AutoModel.from_pretrained(options.model_name, from_tf=True).to(device).eval()
    tokenizer = SimpleBatchTokenizer(AutoTokenizer.from_pretrained(options.tokenizer_name), options.batch_size)
    provider = InnerProductMatrixProvider()
    return UnsupervisedTransformerReranker(model, tokenizer, provider)


def construct_classification_transformer(cls, reranker_cls):
    def construct(options: KaggleEvaluationOptions) -> Reranker:
        try:
            model = cls.from_pretrained(options.model_name)
        except OSError:
            model = cls.from_pretrained(options.model_name, from_tf=True)
        device = torch.device(options.device)
        model = model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(options.tokenizer_name)
        return reranker_cls(model, tokenizer)
    return construct


construct_seq_class_transformer = construct_classification_transformer(AutoModelForSequenceClassification,
                                                                       SequenceClassificationTransformerReranker)


def construct_qa_transformer(options: KaggleEvaluationOptions) -> Reranker:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(options.model_name)
    except OSError:
        model = AutoModelForSequenceClassification.from_pretrained(options.model_name, from_tf=True)
    fixed_model = BertForQuestionAnswering(model.config)
    fixed_model.qa_outputs = model.classifier
    fixed_model.bert = model.bert
    device = torch.device(options.device)
    model = fixed_model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(options.tokenizer_name)
    return QuestionAnsweringTransformerReranker(model, tokenizer)


def construct_bm25(_: KaggleEvaluationOptions) -> Reranker:
    return Bm25Reranker(index_path=SETTINGS.cord19_index_path)


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(opt('--dataset', type=Path, default='data/kaggle-lit-review.json'),
                 opt('--method', required=True, type=str, choices=METHOD_CHOICES),
                 opt('--model-name', type=str),
                 opt('--split', type=str, default='nq', choices=('nq', 'kq')),
                 opt('--batch-size', '-bsz', type=int, default=96),
                 opt('--device', type=str, default='cuda:0'),
                 opt('--tokenizer-name', type=str),
                 opt('--metrics', type=str, nargs='+', default=metric_names(), choices=metric_names()))
    args = apb.parser.parse_args()
    options = KaggleEvaluationOptions(**vars(args))
    ds = LitReviewDataset.from_file(str(options.dataset))
    examples = ds.to_senticized_dataset(SETTINGS.cord19_index_path, split=options.split)
    construct_map = dict(transformer=construct_transformer,
                         bm25=construct_bm25,
                         t5=construct_t5,
                         seq_class_transformer=construct_seq_class_transformer,
                         qa_transformer=construct_qa_transformer)
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
