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
    one_mono_score = collections.OrderedDict()
    min_mono_score = collections.OrderedDict()
    one_duo_score = collections.OrderedDict()
    min_duo_score = collections.OrderedDict()
    last_duo_rank = collections.defaultdict(int)
    seen_queries = set()
    with open(args.run_mono) as f_mono:
        with open(args.run_duo) as f_duo:
            with open(args.run_duo) as f_duo2:
                for lduo in tqdm(f_duo2):
                    query_id, a, doc_title, rank, b, c = lduo.split()
                    last_duo_rank[query_id] = max(last_duo_rank[query_id], int(rank))

            with open(args.output_run, 'w') as f_merge:
                for lmono in tqdm(f_mono):
                    query_id, a, doc_title, rank, b, c = lmono.split()
                    b = float(b)
                    if query_id not in run:
                        run[query_id] = []
                    if int(rank) == 1:
                        one_mono_score[query_id] = b
                    if int(rank) == last_duo_rank[query_id]:
                        min_mono_score[query_id] = b
                    run[query_id].append([a, doc_title, rank, b, "duot5"])
                for lduo in tqdm(f_duo):
                    query_id, a, doc_title, rank, b, c = lduo.split()
                    if int(rank) == last_duo_rank[query_id]:
                        min_duo_score[query_id] = float(b)
                    if int(rank) == 1:
                        one_duo_score[query_id] = float(b)
                with open(args.run_duo) as f_duo2:
                    for lduo in tqdm(f_duo2):
                        query_id, a, doc_title, rank, b, c = lduo.split()
                        score = min_mono_score[query_id] + (((one_mono_score[query_id] - min_mono_score[query_id])/(one_duo_score[query_id] - min_duo_score[query_id])) * (float(b) - min_duo_score[query_id]))
                        run[query_id][int(rank) - 1] = [a, doc_title, rank, score, "duot5"]
                for query_id in run:
                    for i, j in enumerate(run[query_id]):
                        j[-2] = str(j[-2])
                        f_merge.write(f'{query_id} ' + ' '.join(j) + '\n')

