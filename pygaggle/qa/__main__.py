import sys
import argparse
from pygaggle.qa.obqa import OpenBookQA
from pygaggle.qa.dpr_reader import DprReader
from pyserini.search import SimpleSearcher
from pyserini.dsearch import SimpleDenseSearcher, DprQueryEncoder


def define_reader_args(parser):
    parser.add_argument('--model', type=str, required=True, help="Reader model name or path")
    parser.add_argument('--device', type=str, required=False, default='cuda:0', help="Device to run inference on")


def define_retriever_args(parser):
    parser.add_argument('--model', type=str, required=False, help="Retriever query encoder name or path")
    parser.add_argument('--index', type=str, required=True, help="Pyserini index name or path")
    parser.add_argument('--corpus', type=str, required=True, help="Pyserini sparse index name or path, serve as corpus")


def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive QA')

    commands = parser.add_subparsers(title='sub-commands')

    dense_parser = commands.add_parser('reader')
    define_reader_args(dense_parser)

    sparse_parser = commands.add_parser('retriever')
    define_retriever_args(sparse_parser)

    args = parse_args(parser, commands)

    print("Init QA models")
    reader = DprReader(args.reader.model, device=args.reader.device)
    if args.retriever.model:
        retriever = SimpleDenseSearcher(args.retriever.index, DprQueryEncoder(args.retriever.model))
    else:
        retriever = SimpleSearcher.from_prebuilt_index(args.retriever.corpus)
    corpus = SimpleSearcher.from_prebuilt_index(args.retriever.corpus)
    obqa = OpenBookQA(reader, retriever, corpus)

    # run a warm up question
    obqa.predict('what is lobster roll')
    while True:
        question = input('Please enter a question: ')
        answer = obqa.predict(question)
        answer_text = answer["answer"]
        answer_context = answer["context"]["text"]
        print(f"ANSWER:\t {answer_text}")
        print(f"CONTEXT:\t {answer_context}")
