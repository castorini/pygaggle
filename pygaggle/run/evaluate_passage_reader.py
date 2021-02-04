from typing import Optional
from pathlib import Path
import logging
import json
import os
import numpy as np

from pydantic import BaseModel
from transformers import (DPRReader,
                          DPRReaderTokenizer)

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

def display(ems):
    em = np.mean(np.array(ems))
    logging.info(f'Exact Match Accuracy: {em * 100}')

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
            help='File to output predictions for each example; if no output file specified, this output will be discarded'),
        opt('--device',
            type=str,
            default='cuda:0',
            help='Device for model computations'),
    )
    args = apb.parser.parse_args()
    options = PassageReadingEvaluationOptions(**vars(args))

    logging.info("Loading Reader Model and Tokenizer.")
    construct_map = dict(
        dpr=construct_dpr,
    )
    reader = construct_map[options.method](options)

    evaluator = ReaderEvaluator(reader)

    retrievalFile = options.retrieval_file
    if os.path.isfile(retrievalFile):
        files = [retrievalFile]
    else:
        files = os.listdir(retrievalFile)
        files.sort()
        files = map(lambda filename: os.path.join(retrievalFile, filename), files)

    ems = []
    nQueries = 0
    if args.output_file is not None:
        dpr_predictions = []
    else:
        dpr_predictions = None

    for filename in files:
        logging.info(f'Read {nQueries} queries.')
        with open(filename) as f:
            data = json.load(f)

        nQueries += len(data)
        examples = []
        for _, item in data.items():
            examples.append(
                RetrievalExample(
                    query=Query(text=item["question"]),
                    texts=list(map(lambda context: Text(text=context["text"].split('\n', 1)[1], title=context["text"].split('\n', 1)[0][1:-1]), item["contexts"]))[:options.use_top_k_passages],
                    groundTruthAnswers=item["answers"],
                )
            )

        ems.extend(evaluator.evaluate(examples, dpr_predictions))
        display(ems)

    logging.info(f'Reader completed.\nRead {nQueries} queries.')
    display(ems)

    if args.output_file is not None:
        with open(args.output_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(dpr_predictions, f, indent=4)


if __name__ == '__main__':
    main()
