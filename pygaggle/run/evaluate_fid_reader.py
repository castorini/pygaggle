from pygaggle.qa.fid_reader import FidReader
from typing import List, Optional
from pathlib import Path
import logging
import json
import numpy as np
import subprocess

from pydantic import BaseModel

from .args import ArgumentParserBuilder, opt
from pygaggle.qa.base import Reader, Question, Context
from pygaggle.qa.dpr_reader import DprReader
from pygaggle.qa.span_selection import DprSelection, GarSelection, DprFusionSelection, GarFusionSelection
from pygaggle.data.retrieval import RetrievalExample
from typing import List, Optional, Dict
from tqdm import tqdm
import numpy as np
import string
import regex as re


class ReaderEvaluator:
    """Class for evaluating a reader.
    Takes in a list of examples (query, texts, ground truth answers),
    predicts a list of answers using the Reader passed in, and
    collects the exact match accuracies between the best answer and
    the ground truth answers given in the example.
    Exact match scoring used is identical to the DPR repository.
    """

    def __init__(
        self,
        reader: Reader,
    ):
        self.reader = reader

    def evaluate(
        self,
        examples: List[RetrievalExample],
        topk_em: List[int] = [50],
        dpr_predictions: Optional[Dict[int, List[Dict[str, str]]]] = None,
    ):
        ems = {str(setting): {k: [] for k in topk_em} for setting in self.reader.span_selection_rules}
        for example in tqdm(examples):
            answers = self.reader.predict(example.question, example.contexts, topk_em)

        return ems

    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return ReaderEvaluator._normalize_answer(prediction) == ReaderEvaluator._normalize_answer(ground_truth)

    @staticmethod
    def _normalize_answer(s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


class PassageReadingEvaluationOptions(BaseModel):
    task: str
    retriever: str
    reader: str
    settings: List[str]
    retrieval_file: Path
    model_name: Optional[str]
    tokenizer_name: Optional[str]
    num_spans: int
    max_answer_length: int
    num_spans_per_passage: int
    text_maxlength: int
    device: str
    batch_size: int
    topk_em: List[int]

def construct_fid(options: PassageReadingEvaluationOptions) -> Reader:
    model = options.model_name
    tokenizer = options.tokenizer_name

    span_selection_rules = [parse_span_selection_rules(setting) for setting in options.settings]
    return FidReader(model,
                     tokenizer,
                     span_selection_rules,
                     options.num_spans,
                     options.max_answer_length,
                     options.num_spans_per_passage,
                     options.text_maxlength,
                     options.batch_size,
                     options.device)



def parse_span_selection_rules(settings):
    settings = settings.split('_')

    settings_map = dict(
        dpr=DprSelection,
        dprfusion=DprFusionSelection,
        gar=GarSelection,
        garfusion=GarFusionSelection,
    )
    return settings_map[settings[0]](*settings[1:])

def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(
        opt('--task',
            type=str,
            default='wikipedia'),
        opt('--retriever',
            type=str,
            default='score',
            help='Retriever score field to rank the input passages to the reader'),
        opt('--reader',
            type=str,
            default='fid'),
        opt('--settings',
            type=str,
            nargs='+',
            default=['dpr']),
        opt('--retrieval-file',
            type=Path,
            required=True,
            help='JSON file containing top passages selected by the retrieval model'),
        opt('--model-name',
            type=str,
            required=True,
            default='nq_reader_base',
            help='Pretrained model for reader'),
        opt('--tokenizer-name',
            type=str,
            default='t5-base',
            help='Pretrained model for tokenizer'),
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
            required=True,
            help='File to output predictions for each example; if not specified, this output will be discarded'),
        opt('--text_maxlength',
            type=int,
            default=250,
            required=False,
            help='maximum number of tokens in text segments (question+passage)'),
        opt('--device',
            type=str,
            default='cuda:0',
            help='Device for model computations'),
        opt('--batch-size',
            type=int,
            default=16,
            help='batch size of reader inference'),
        opt('--topk-retrieval',
            type=int,
            default=[],
            nargs='+',
            help='Values of k to print the topk accuracy of the retrieval file'),
        opt('--topk-em',
            type=int,
            default=[100],
            nargs='+',
            help='Values of k to print the topk exact match score'),
    )
    args = apb.parser.parse_args()
    options = PassageReadingEvaluationOptions(**vars(args))

    logging.info('Loading the Retrieval File')
    with open(options.retrieval_file) as f:
        data = json.load(f)

    if args.topk_retrieval:
        logging.info('Evaluating Topk Retrieval Accuracies')
        subprocess.call(['python',
                         'tools/scripts/dpr/evaluate_retrieval.py',
                         '--retrieval',
                         options.retrieval_file,
                         '--topk',
                         *map(str, args.topk_retrieval)])

    logging.info('Loading Reader Model and Tokenizer')
    construct_map = dict(
        # dpr=construct_dpr,
        fid=construct_fid
    )
    reader = construct_map[options.reader](options)

    evaluator = ReaderEvaluator(reader)

    max_topk_passages = max(options.topk_em)
    examples = []
    for _, item in data.items():
        topk_contexts = sorted(item['contexts'], reverse=True, key=lambda context: float(context[options.retriever]))[
                        : max_topk_passages]
        texts = list(map(lambda context: Context(text=context['text'].split('\n', 1)[1].replace('""', '"'),
                                                 title=context['text'].split('\n', 1)[0].replace('"', ''),
                                                 score=float(context[options.retriever])),
                         topk_contexts))
        examples.append(
            RetrievalExample(
                question=Question(text=item['question']),
                contexts=texts,
                ground_truth_answers=item['answers'],
            )
        )

    from tqdm import tqdm
    results = []
    scores = []
    for example in tqdm(examples):
        answer = reader.predict(example.question, example.contexts, options.topk_em)
        results.append({"question": example.question.text,
                        "answers": example.ground_truth_answers,
                        'prediction': answer})
        score = max([ReaderEvaluator.exact_match_score(answer, ga) for ga in example.ground_truth_answers])
        scores.append(score)
    
    logging.info('prediction completed')
    em = np.mean(np.array(scores)) * 100.
    logging.info(f'Exact Match Accuracy: {em}')

    with open(args.output_file, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    main()
