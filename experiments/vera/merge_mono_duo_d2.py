"""Script to evaluate search engine on TREC-covid qrels."""
import argparse
import logging
import collections
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Retrieve documents for TREC-COVID queries.')
    parser.add_argument('--run_mono', required=True, default='',
                        help='TREC mono run filee')
    parser.add_argument('--run_duo', required=True, default='',
                        help='TREC mono run filee')
    parser.add_argument('--output_run', required=True, default='',
                        help='output filee')
    parser.add_argument('--max_docs_run', type=int, default=50,
                        help='number of hits to write to the run file.')

    args = parser.parse_args()

    logging.info(args)

    run = collections.OrderedDict()
    seen_queries = set()
    with open(args.run_mono) as f_mono:
        with open(args.run_duo) as f_duo:
            with open(args.output_run, 'w') as f_merge:
                for lmono in tqdm(f_mono):
                    query_id, a, doc_title, rank, b, c = lmono.split(' ')
                    if query_id not in run:
                        run[query_id] = []
                    run[query_id].append([a, doc_title, rank, float(b), "h2oloo.m11"])
                for lduo in tqdm(f_duo):
                    query_id, a, doc_title, rank, b, c = lduo.split(' ')
                    run[query_id][int(rank) - 1] = [a, doc_title, rank, float(b) + run[query_id][int(rank) - 1][3], "h2oloo.m11"]
                for query_id in run:
                    for i, j in enumerate(sorted(run[query_id], key=lambda ex: ex[3], reverse=True)):
                        j[-2] = str(j[-2])
                        j[-3] = str(i + 1)
                        f_merge.write(f'{query_id} ' + ' '.join(j) + '\n')

