from typing import Optional, List
from pathlib import Path
import logging

from pydantic import BaseModel, validator
from transformers import AutoModel, AutoTokenizer
import torch

from .args import ArgumentParserBuilder, opt
from pygaggle.rerank import TransformerReranker, InnerProductMatrixProvider, Reranker, T5Reranker, TfIdfReranker
from pygaggle.model import SimpleBatchTokenizer, CachedT5ModelLoader, T5BatchTokenizer, SpacyWordTokenizer, \
    RerankerEvaluator, metric_names
from pygaggle.data import LitReviewDataset
from pygaggle.settings import Settings


SETTINGS = Settings()
METHOD_CHOICES = ('transformer', 'tfidf', 't5')


class KaggleEvaluationOptions(BaseModel):
    dataset: Path
    method: str
    batch_size: int
    device: str
    metrics: List[str]
    model_name: Optional[str]
    word_tokenizer: Optional[str]

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

    @validator('word_tokenizer')
    def word_tokenizer_sane(cls, v: Optional[str], values, **kwargs):
        if values['method'] == 'tfidf' and v is None:
            raise ValueError('word tokenizer must be specified for tf-idf')
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
    model = AutoModel.from_pretrained(options.model_name).to(device).eval()
    tokenizer = SimpleBatchTokenizer(AutoTokenizer.from_pretrained(options.model_name), options.batch_size)
    provider = InnerProductMatrixProvider()
    return TransformerReranker(model, tokenizer, provider)


def construct_tfidf(_: KaggleEvaluationOptions) -> Reranker:
    return TfIdfReranker(SpacyWordTokenizer())


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(opt('--dataset', type=Path, default='data/kaggle-lit-review.json'),
                 opt('--method', required=True, type=str, choices=METHOD_CHOICES),
                 opt('--model-name', type=str),
                 opt('--batch-size', '-bsz', type=int, default=96),
                 opt('--device', type=str, default='cuda:0'),
                 opt('--word-tokenizer', '-wt', type=str, default='spacy', choices=('spacy',)),
                 opt('--metrics', type=str, nargs='+', default=metric_names(), choices=metric_names()))
    args = apb.parser.parse_args()

    options = KaggleEvaluationOptions(**vars(args))
    ds = LitReviewDataset.from_file(str(options.dataset))
    examples = ds.to_senticized_dataset(SETTINGS.cord19_index_path)
    construct_map = dict(transformer=construct_transformer, tfidf=construct_tfidf, t5=construct_t5)
    reranker = construct_map[options.method](options)
    evaluator = RerankerEvaluator(reranker, options.metrics)
    width = max(map(len, args.metrics)) + 1
    stdout = []
    for metric in evaluator.evaluate(examples):
        logging.info(f'{metric.name:<{width}}{metric.value:.5}')
        stdout.append(f'{metric.name.title()}\t{metric.value}')
    print('\n'.join(stdout))


if __name__ == '__main__':
    main()
