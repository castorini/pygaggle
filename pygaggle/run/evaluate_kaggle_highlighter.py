from typing import Optional, List
from pathlib import Path
import logging

from pydantic import BaseModel, validator
from transformers import (
                          AutoModel,
                          AutoModelForQuestionAnswering,
                          AutoTokenizer,
                          BertForSequenceClassification,
                         )
import torch

from .args import ArgumentParserBuilder, opt
from pygaggle.rerank.base import Reranker
from pygaggle.rerank.bm25 import Bm25Reranker
from pygaggle.rerank.transformer import (
    QuestionAnsweringTransformerReranker,
    MonoBERT,
    MonoT5,
    UnsupervisedTransformerReranker,
    SentenceTransformersReranker
    )
from pygaggle.rerank.random import RandomReranker
from pygaggle.rerank.similarity import CosineSimilarityMatrixProvider
from pygaggle.model import (RerankerEvaluator,
                            SimpleBatchTokenizer,
                            metric_names)
from pygaggle.data import LitReviewDataset
from pygaggle.settings import Cord19Settings


SETTINGS = Cord19Settings()
METHOD_CHOICES = ('transformer', 'minilm', 'bm25', 't5', 'seq_class_transformer',
                  'qa_transformer', 'random')


class KaggleEvaluationOptions(BaseModel):
    dataset: Path
    index_dir: Path
    method: str
    batch_size: int
    device: str
    split: str
    do_lower_case: bool
    metrics: List[str]
    model: Optional[str]
    tokenizer_name: Optional[str]

    @validator('dataset')
    def dataset_exists(cls, v: Path):
        assert v.exists(), 'dataset must exist'
        return v

    @validator('model')
    def model_sane(cls, v: Optional[str], values, **kwargs):
        method = values['method']
        if method == 'transformer' and v is None:
            raise ValueError('transformer name must be specified')
        if v == 'biobert':
            return 'monologg/biobert_v1.1_pubmed'
        return v

    @validator('tokenizer_name')
    def tokenizer_sane(cls, v: str, values, **kwargs):
        if v is None:
            return values['model']
        return v


def construct_t5(options: KaggleEvaluationOptions) -> Reranker:
    model = MonoT5.get_model(options.model,
                             device=options.device)
    tokenizer = MonoT5.get_tokenizer(options.model, batch_size=options.batch_size)
    return MonoT5(model = model, tokenizer = tokenizer)


def construct_transformer(options: KaggleEvaluationOptions) -> Reranker:
    device = torch.device(options.device)
    try:
        model = AutoModel.from_pretrained(options.model).to(device).eval()
    except OSError:
        model = AutoModel.from_pretrained(options.model,
                                          from_tf=True).to(device).eval()
    tokenizer = SimpleBatchTokenizer(
                    AutoTokenizer.from_pretrained(
                        options.tokenizer_name, do_lower_case=options.do_lower_case),
                    options.batch_size)
    provider = CosineSimilarityMatrixProvider()
    return UnsupervisedTransformerReranker(model, tokenizer, provider)


def construct_seq_class_transformer(options:
                                    KaggleEvaluationOptions) -> Reranker:
    try:
        model = MonoBERT.get_model(options.model, device=options.device)
    except OSError:
        try:
            model = MonoBERT.get_model(
                        options.model,
                        from_tf=True,
                        device=options.device)
        except AttributeError:
            # Hotfix for BioBERT MS MARCO. Refactor.
            BertForSequenceClassification.bias = torch.nn.Parameter(
                                                    torch.zeros(2))
            BertForSequenceClassification.weight = torch.nn.Parameter(
                                                    torch.zeros(2, 768))
            model = BertForSequenceClassification.from_pretrained(
                        options.model, from_tf=True)
            model.classifier.weight = BertForSequenceClassification.weight
            model.classifier.bias = BertForSequenceClassification.bias
            device = torch.device(options.device)
            model = model.to(device).eval()
    tokenizer = MonoBERT.get_tokenizer(
                    options.tokenizer_name, do_lower_case=options.do_lower_case)
    return MonoBERT(model, tokenizer)


def construct_qa_transformer(options: KaggleEvaluationOptions) -> Reranker:
    # We load a sequence classification model first -- again, as a workaround.
    # Refactor
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(
                    options.model)
    except OSError:
        model = AutoModelForQuestionAnswering.from_pretrained(
                    options.model, from_tf=True)
    device = torch.device(options.device)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
                    options.tokenizer_name,
                    do_lower_case=options.do_lower_case,
                    use_fast=False)
    return QuestionAnsweringTransformerReranker(model, tokenizer)

  
def construct_minilm(options: KaggleEvaluationOptions) -> Reranker:
    if options.model:
        return SentenceTransformersReranker(options.model, use_amp=True)
    else:
        return SentenceTransformersReranker(use_amp=True)


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
                 opt('--model',
                     type=str,
                     help='Path to pre-trained model or huggingface model name'),
                 opt('--split', type=str, default='nq', choices=('nq', 'kq')),
                 opt('--batch-size', '-bsz', type=int, default=96),
                 opt('--device', type=str, default='cuda:0'),
                 opt('--tokenizer-name', type=str),
                 opt('--do-lower-case', action='store_true'),
                 opt('--metrics',
                     type=str,
                     nargs='+',
                     default=metric_names(),
                     choices=metric_names()),
                 opt('--model-type', type=str))
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
                         minilm=construct_minilm,
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
