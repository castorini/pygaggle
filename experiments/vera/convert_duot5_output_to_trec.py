import argparse
import collections
import numpy as np


parser = argparse.ArgumentParser(
    description='Convert T5 predictions into a TREC-formatted run.')
parser.add_argument('--predictions', type=str, required=True, help='T5 predictions file.')
parser.add_argument('--query_run_ids', type=str, required=True,
                    help='File containing query doc id pairs paired with the T5\'s predictions file.')
parser.add_argument('--output', type=str, required=True, help='run file in the TREC format.')

args = parser.parse_args()

examples = collections.defaultdict(dict)
with open(args.query_run_ids) as f_query_run_ids, open(args.predictions) as f_pred:
    for line_query_doc_id, line_pred in zip(f_query_run_ids, f_pred):
        query_id, doc_id_a, doc_id_b = line_query_doc_id.strip().split()
        doc_id_a = doc_id_a.split("#")[0]
        doc_id_b = doc_id_b.split("#")[0]
        _, score = line_pred.strip().split()
        score = float(score)
        if doc_id_a not in examples[query_id]:
            examples[query_id][doc_id_a] = 0
        if doc_id_b not in examples[query_id]:
            examples[query_id][doc_id_b] = 0
        examples[query_id][doc_id_a] += np.exp(score)
        examples[query_id][doc_id_b] += 1 - np.exp(score)

with open(args.output, 'w') as fout:
    for query_id, doc_ids_scores in examples.items():
        doc_ids_scores = [
            (doc_id, scores)
            for doc_id, scores in doc_ids_scores.items()]

        doc_ids_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(doc_ids_scores):
            print(2*(len(doc_ids_scores) - 1))
            fout.write(
                f'{query_id} Q0 {doc_id} {rank + 1} {score/(2*(len(doc_ids_scores) - 1))} duot5\n')

