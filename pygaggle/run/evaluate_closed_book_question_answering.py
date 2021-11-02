from pathlib import Path
import logging
import json
import numpy as np

from pydantic import BaseModel
from tqdm import tqdm

from .args import ArgumentParserBuilder, opt
from pygaggle.qa.cbqa import ClosedBookQA
from pygaggle.model.evaluate import ReaderEvaluator


class ClosedBookQuestionAnsweringEvaluationOptions(BaseModel):
    data: Path
    model_name: str
    device: str


def main():
    apb = ArgumentParserBuilder()
    apb.add_opts(
        opt('--data',
            type=Path,
            help='Path to the dataset to run the model on'),
        opt('--model-name',
            type=str,
            default='google/t5-large-ssm-nq',
            help='Pretrained model for closed book question answering'),
        opt('--output-file',
            type=Path,
            default=None,
            help='File to output predictions for each example; if not specified, this output will be discarded'),
        opt('--device',
            type=str,
            default='cuda:0',
            help='Device for model computations'),
    )
    args = apb.parser.parse_args()
    options = ClosedBookQuestionAnsweringEvaluationOptions(**vars(args))

    with open(options.data) as f:
        data = json.load(f)

    logging.info('Loading CBQA Model and Tokenizer')

    cbqa = ClosedBookQA(options.model_name, options.device)
    results = []
    scores = []
    for _, item in tqdm(data.items()):
        prediction = cbqa.predict(item['question'])
        if args.output_file is not None:
            results.append({"question": item['question'],
                            "answers": item['answers'],
                            'prediction': prediction})
        scores.append(max([ReaderEvaluator.exact_match_score(prediction, ga) for ga in item['answers']]))

    logging.info('CBQA prediction completed')
    em = np.mean(np.array(scores)) * 100.
    logging.info(f'Exact Match Accuracy: {em}')

    if args.output_file is not None:
        with open(args.output_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
