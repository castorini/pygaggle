import sys
import argparse
from pygaggle.qa.cbqa import ClosedBookQA
from pygaggle.qa.obqa import OpenBookQA
from pygaggle.qa.dpr_reader import DprReader
from pyserini.search import SimpleSearcher
from pyserini.dsearch import SimpleDenseSearcher, DprQueryEncoder


def arg_check(args, parser):
    if args.type == 'openbook':
        requirements = ['retriever_index', 'retriever_corpus', 'reader_model']
    elif args.type == 'closedbook':
        requirements = ['cbqa_model']

    for requirement in requirements:
        if getattr(args, requirement) is None:
            parser.error(f"--{requirement.replace('_', '-')} required when using {args.type} qa!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive QA')
    parser.add_argument('--type', type=str, choices=["openbook", "closedbook"], default='openbook', help="Open-book or closed-book question answering.")

    parser.add_argument('--cbqa-model', type=str, required=False, help="CBQA model name or path")
    parser.add_argument('--cbqa-device', type=str, required=False, default='cuda:0', help="Device to run inference on")

    parser.add_argument('--retriever-model', type=str, required=False, help="Retriever query encoder name or path")
    parser.add_argument('--retriever-index', type=str, required=False, help="Pyserini index name or path")
    parser.add_argument('--retriever-corpus', type=str, required=False, help="Pyserini sparse index name or path, serve as corpus")
    # index corpus, deivce
    parser.add_argument('--reader-model', type=str, required=False, help="Reader model name or path")
    parser.add_argument('--reader-device', type=str, required=False, default='cuda:0', help="Device to run inference on")

    args = parser.parse_args()

    # check arguments
    arg_check(args, parser)

    print("Init QA models")
    if args.type == 'openbook':
        reader = DprReader(args.reader_model, device=args.reader_device)
        if args.retriever_model:
            retriever = SimpleDenseSearcher(args.retriever_index, DprQueryEncoder(args.retriever_model))
        else:
            retriever = SimpleSearcher.from_prebuilt_index(args.retriever_corpus)
        corpus = SimpleSearcher.from_prebuilt_index(args.retriever_corpus)
        obqa = OpenBookQA(reader, retriever, corpus)
        # run a warm up question
        obqa.predict('what is lobster roll')
        while True:
            question = input('Please enter a question: ')
            answer = obqa.predict(question)
            answer_text = answer["answer"]
            answer_context = answer["context"]["text"]
            print(f"Answer:\t {answer_text}")
            print(f"Context:\t {answer_context}")
    else:
        cbqa = ClosedBookQA(args.cbqa_model, args.cbqa_device)
        # run a warm up question
        cbqa.predict('what is lobster roll')
        while True:
            question = input('Please enter a question: ')
            answer = cbqa.predict(question)
            print(f"Answer:\t {answer}")

