"""Script to evaluate search engine on TREC-covid qrels."""
import argparse
import logging
import collections
from tqdm import tqdm
import numpy as np

def load_run(path, topk=1000):
    """Loads run into a dict of key: query_id, value: list of candidate doc
    ids."""

    # We want to preserve the order of runs so we can pair the run file with
    # the TFRecord file.
    print('Loading run...')
    run = collections.OrderedDict()
    with open(path) as f:
        for line in tqdm(f):
            query_id, _, doc_title, rank, score, _ = line.split()
            if query_id not in run:
                run[query_id] = {}
            run[query_id][doc_title] = float(score)

    return run
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Retrieve documents for TREC-COVID queries.')
    parser.add_argument('--run_a', required=True, default='',
                        help='TREC mono run filee')
    parser.add_argument('--run_b', required=True, default='',
                        help='TREC mono run filee')
    parser.add_argument('--output', required=True, default='',
                        help='output filee')
    parser.add_argument('--alpha', required=True, type=float)

    args = parser.parse_args()

    run_a = load_run(args.run_a)
    run_b = load_run(args.run_b)
    run = collections.OrderedDict()
    for query_id in run_a:
        run[query_id] = []
        for doc_id in run_a[query_id]:
            run[query_id].append((doc_id, args.alpha * run_a[query_id][doc_id] + (1.0 - args.alpha) * run_b[query_id][doc_id]))

    with open(args.output, 'w') as fout:
        for query_id, doc_ids_scores in run.items():
            doc_ids_scores.sort(key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(doc_ids_scores):
                fout.write(f'{query_id} Q0 {doc_id} {rank + 1} {score} vera{args.alpha}\n')

    print('Done!')
