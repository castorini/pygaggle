from typing import Optional, List
from pathlib import Path
import logging

from pydantic import BaseModel, validator
from transformers import (AutoModel,
                          AutoTokenizer,
                          AutoModelForSequenceClassification,
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
                            TrecWriter)
from pygaggle.data import TRECCovidDataset
from pygaggle.settings import TRECCovidSettings


SETTINGS = TRECCovidSettings()
METHOD_CHOICES = ('transformer', 'bm25', 't5', 'seq_class_transformer',
                  'random')


class DocumentRankingEvaluationOptions(BaseModel):
    task: str
    dataset: Path
    index_dir: Path
    method: str
    model: str
    batch_size: int
    seg_size: int
    seg_stride: int
    aggregate_method: str
    device: str
    from_tf: bool
    metrics: List[str]
    model_type: Optional[str]
    tokenizer_name: Optional[str]

    @validator('task')
    def task_exists(cls, v: str):
        assert v in ['trec-covid']

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
    device = torch.device(options.device)
    model = T5ForConditionalGeneration.from_pretrained(options.model,
                                                       from_tf=options.from_tf).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(options.model_type, use_fast=False)
    tokenizer = T5BatchTokenizer(tokenizer, options.batch_size)
    return T5Reranker(model, tokenizer)


def construct_transformer(options:
                          DocumentRankingEvaluationOptions) -> Reranker:
    device = torch.device(options.device)
    model = AutoModel.from_pretrained(options.model,
                                      from_tf=options.from_tf).to(device).eval()
    tokenizer = SimpleBatchTokenizer(AutoTokenizer.from_pretrained(
        options.tokenizer_name, use_fast=False),
        options.batch_size)
    provider = CosineSimilarityMatrixProvider()
    return UnsupervisedTransformerReranker(model, tokenizer, provider)


def construct_seq_class_transformer(options: DocumentRankingEvaluationOptions
                                    ) -> Reranker:
    model = AutoModelForSequenceClassification.from_pretrained(options.model, from_tf=options.from_tf)
    device = torch.device(options.device)
    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(options.tokenizer_name, use_fast=False)
    return SequenceClassificationTransformerReranker(model, tokenizer)


def construct_bm25(options: DocumentRankingEvaluationOptions) -> Reranker:
    return Bm25Reranker(index_path=str(options.index_dir))


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(opt('--task',
                     type=str,
                     default='trec-covid',
                     help='A string correspondding to the task to execute. By default, this is "trec-covid".'),
                 opt('--dataset',
                     type=Path,
                     required=True,
                     help='Path to the directory containing the topics file, qrels file, and run file.'),
                 opt('--index-dir',
                     type=Path,
                     required=True,
                     help='Path to the input Anserini index.'),
                 opt('--method',
                     required=True,
                     type=str,
                     choices=METHOD_CHOICES,
                     help='Specifies the type of reranker to use.'),
                 opt('--model',
                     required=True,
                     type=str,
                     help='Path to pre-trained model or huggingface model name.'),
                 opt('--output-file',
                     type=Path,
                     default='.',
                     help='A path to the output file.'),
                 opt('--overwrite-output',
                     action='store_true',
                     help='If set to true, output will be overwritten if the output file already exists. Otherwise, '
                          'output will be appended to the existing file.'),
                 opt('--batch-size',
                     '-bsz',
                     type=int,
                     default=96,
                     help='The batch size for tokenization.'),
                 opt('--device',
                     type=str,
                     default='cuda:0',
                     help='The CUDA device to use for reranking.'),
                 opt('--from-tf',
                     action='store_true',
                     help='A boolean of whether the pretrained model is being loaded from a Tensorflow checkpoint. '
                          'If flag is unused, assumed to be false.'),
                 opt('--metrics',
                     type=str,
                     nargs='+',
                     default=metric_names(),
                     choices=metric_names(),
                     help='The list of metrics to collect while evaluating the reranker.'),
                 opt('--model-type',
                     type=str,
                     help='The T5 tokenizer name.'),
                 opt('--tokenizer-name',
                     type=str,
                     help='The name of the tokenizer to pull from huggingface using the AutoTokenizer class. If '
                     'left empty, this will be set to the model name.'),
                 opt('--seg-size',
                     type=int,
                     default=10,
                     help='The number of sentences in each segment. For example, given a document with sentences'
                     '[1,2,3,4,5], a seg_size of 3, and a stride of 2, the document will be broken into segments'
                     '[[1, 2, 3], [3, 4, 5], and [5]].'),
                 opt('--seg-stride',
                     type=int,
                     default=5,
                     help='The number of sentences to increment between each segment. For example, given a document'
                     'with sentences [1,2,3,4,5], a seg_size of 3, and a stride of 2, the document will be broken into'
                     'segments [[1, 2, 3], [3, 4, 5], and [5]].'),
                 opt('--aggregate-method',
                     type=str,
                     default="max",
                     help='Aggregation method for combining scores across sentence segments of the same document.'))
    args = apb.parser.parse_args()
    options = DocumentRankingEvaluationOptions(**vars(args))
    logging.info("Preprocessing Queries & Docs:")
    ds = TRECCovidDataset.from_folder(str(options.dataset))
    examples = ds.to_relevance_examples(str(options.index_dir))
    logging.info("Loading Ranker & Tokenizer:")
    construct_map = dict(transformer=construct_transformer,
                         bm25=construct_bm25,
                         t5=construct_t5,
                         seq_class_transformer=construct_seq_class_transformer,
                         random=lambda _: RandomReranker())
    # Retrieve the correct reranker from the options map based on the input flag.
    reranker = construct_map[options.method](options)
    writer = TrecWriter(args.output_file, args.overwrite_output)
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
