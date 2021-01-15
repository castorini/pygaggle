"""
This script convert duoT5 output file to msmarco run file
"""
import argparse
import collections
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--t5_output", type=str, required=True,
                    help="tsv file with two columns, <label> and <score>")
parser.add_argument("--t5_output_ids", type=str, required=True,
                    help="tsv file with five columns <query_id>, <doc_id_a>, <doc_id_b>, <rank_a> and <rank_b>")
parser.add_argument("--input_run", type=str, required=True,
                    help="path to input run, tsv file, with <query_id>, <doc_id> and <rank>")
parser.add_argument("--duo_run", type=str, required=True,
                    help="path to output duo run, tsv file, with <query_id>, <doc_id> and <rank>")
parser.add_argument("--top_k", type=int, default=50,
                    help="top-k pointwise hits to be reranked by pairwise ranker")
parser.add_argument("--aggregate", type=str, default="sym_sum",
                    help="aggregation technique: one of sum, sym_sum, log_sum or sym_log_sum")

args = parser.parse_args()


def load_run(path):
    """Loads run into a dict of key: query_id, value: list of candidate doc
    ids."""

    # We want to preserve the order of runs so we can pair the run file with
    # the TFRecord file.
    print('Loading run...')
    run = collections.OrderedDict()
    with open(path) as f:
        for line in tqdm(f):
            query_id, doc_title, rank = line.split('\t')
            if query_id not in run:
                run[query_id] = []
            run[query_id].append((doc_title, int(rank)))

    # Sort candidate docs by rank.
    print('Sorting candidate docs by rank...')
    sorted_run = collections.OrderedDict()
    for query_id, doc_titles_ranks in tqdm(run.items()):
        sorted(doc_titles_ranks, key=lambda x: x[1])
        doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
        sorted_run[query_id] = doc_titles

    return sorted_run


input_run = load_run(path=args.input_run)
examples = collections.defaultdict(dict)
with open(args.t5_output_ids) as f_gt, open(args.t5_output) as f_pred:
    for line_gt, line_pred in zip(f_gt, f_pred):
        query_id, doc_id_a, doc_id_b, ct_a, ct_b = line_gt.strip().split('\t')
        _, score = line_pred.strip().split('\t')
        score = float(score)
        if int(ct_a) < args.top_k and int(ct_b) < args.top_k:
            if doc_id_a not in examples[query_id]:
                examples[query_id][doc_id_a] = 0
            if "log" not in args.aggregate:
                score = np.exp(score)
            examples[query_id][doc_id_a] += score
            if "sym" in args.aggregate:
                if doc_id_b not in examples[query_id]:
                    examples[query_id][doc_id_b] = 0
                if "log" in args.aggregate:
                    score_b = np.log(1 - np.exp(score))
                else:
                    score_b = 1 - score
                examples[query_id][doc_id_b] += score_b

for qid in examples:
    examples[qid] = list(examples[qid].items())

with open(args.duo_run, 'w') as fout:
    for query_id, doc_ids_scores in examples.items():
        doc_ids_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(doc_ids_scores):
            fout.write(f'{query_id}\t{doc_id}\t{rank + 1}\n')
        input_offset = len(doc_ids_scores)
        for rank, doc_id in enumerate(input_run[query_id]):
            if rank < input_offset:
                continue
            fout.write(f'{query_id}\t{doc_id}\t{rank + 1}\n')
