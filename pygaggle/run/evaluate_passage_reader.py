from typing import Optional
from pathlib import Path
import logging
import json
import os
import numpy as np
import subprocess

from pydantic import BaseModel

from .args import ArgumentParserBuilder, opt
from pygaggle.reader.base import Reader
from pygaggle.reader.dpr_reader import DensePassageRetrieverReader
from pygaggle.data.retrieval import RetrievalExample
from pygaggle.rerank.base import Query, Text
from pygaggle.model.evaluate import ReaderEvaluator

METHOD_CHOICES = ('dpr')


class PassageReadingEvaluationOptions(BaseModel):
    task: str
    method: str
    retrieval_file: Path
    model_name: Optional[str]
    tokenizer_name: Optional[str]
    use_top_k_passages: int
    num_spans: int
    max_answer_length: int
    num_spans_per_passage: int
    device: str


def construct_dpr(options: PassageReadingEvaluationOptions) -> Reader:
    model = DensePassageRetrieverReader.get_model(options.model_name, options.device)
    tokenizer = DensePassageRetrieverReader.get_tokenizer(options.tokenizer_name)

    return DensePassageRetrieverReader(model,
                                       tokenizer,
                                       options.num_spans,
                                       options.max_answer_length,
                                       options.num_spans_per_passage)


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(
        opt('--task',
            type=str,
            default='wikipedia'),
        opt('--method',
            type=str,
            required=True,
            choices=METHOD_CHOICES),
        opt('--retrieval-file',
            type=Path,
            required=True,
            help='JSON file containing top passages selected by the retrieval model'),
        opt('--model-name',
            type=str,
            default='facebook/dpr-reader-single-nq-base',
            help='Pretrained model for reader'),
        opt('--tokenizer-name',
            type=str,
            default='facebook/dpr-reader-single-nq-base',
            help='Pretrained model for tokenizer'),
        opt('--use-top-k-passages',
            type=int,
            default=50,
            help='The top k passages by the retriever will be used by the reader'),
        opt('--num-spans',
            type=int,
            default=1,
            help='Number of answer spans to return'),
        opt('--max-answer-length',
            type=int,
            default=10,
            help='Maximum length that an answer span can be'),
        opt('--num-spans-per-passage',
            type=int,
            default=10,
            help='Maximum number of answer spans to return per passage'),
        opt('--output-file',
            type=Path,
            default=None,
            help='File to output predictions for each example; if not specified, this output will be discarded'),
        opt('--device',
            type=str,
            default='cuda:0',
            help='Device for model computations'),
        opt('--topk-retrieval',
            type=int,
            default=[],
            nargs='+',
            help='Values of k to print the topk accuracy of the retrieval file'),
    )
    args = apb.parser.parse_args()
    options = PassageReadingEvaluationOptions(**vars(args))

    logging.info("Loading the Retrieval File")
    with open(options.retrieval_file) as f:
        data = json.load(f)

    if args.topk_retrieval:
        logging.info("Evaluating Topk Accuracies")
        subprocess.call(['python',
                         'tools/scripts/dpr/evaluate_retrieval.py',
                         '--retrieval',
                         options.retrieval_file,
                         '--topk',
                         *map(str, args.topk_retrieval)])

    logging.info("Loading Reader Model and Tokenizer")
    construct_map = dict(
        dpr=construct_dpr,
    )
    reader = construct_map[options.method](options)

    evaluator = ReaderEvaluator(reader)

    examples = []
    for _, item in data.items():
        examples.append(
            RetrievalExample(
                query=Query(text=item["question"]),
                texts=list(map(lambda context: Text(text=context["text"].split('\n', 1)[1].replace('""', '"'),
                                                    title=context["text"].split('\n', 1)[0].replace('"', '')),
                               item["contexts"]))[:options.use_top_k_passages],
                ground_truth_answers=item["answers"],
            )
        )
    dpr_predictions = [] if args.output_file is not None else None

    ems = evaluator.evaluate(examples, dpr_predictions)

    logging.info(f'Reader completed')

    em = np.mean(np.array(ems)) * 100.
    logging.info(f'Exact Match Accuracy: {em}')

    if args.output_file is not None:
        with open(args.output_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(dpr_predictions, f, indent=4)


if __name__ == '__main__':
    main()
