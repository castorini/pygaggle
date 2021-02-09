"""Script to convert T5 predictions into a TREC-formatted run."""
import argparse
import collections


parser = argparse.ArgumentParser(
    description='Convert T5 predictions into a TREC-formatted run.')
parser.add_argument('--predictions', type=str, required=True, help='T5 predictions file.')
parser.add_argument('--query_run_ids', type=str, required=True,
                    help='File containing query doc id pairs paired with the T5\'s predictions file.')
parser.add_argument('--output', type=str, required=True, help='run file in the TREC format.')

args = parser.parse_args()

examples = collections.defaultdict(lambda: collections.defaultdict(list))
with open(args.query_run_ids) as f_query_run_ids, open(args.predictions) as f_pred:
    for line_query_doc_id, line_pred in zip(f_query_run_ids, f_pred):
        query_id, doc_id = line_query_doc_id.strip().split('\t')
        _, score = line_pred.strip().split('\t')
        examples[query_id][doc_id].append(float(score))

with open(args.output, 'w') as fout:
    for query_id, doc_ids_scores in examples.items():
        doc_ids_scores = [(doc_id, max(scores)) for doc_id, scores in doc_ids_scores.items()]
        doc_ids_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(doc_ids_scores):
            fout.write(f'{query_id} Q0 {doc_id} {rank + 1} {1 / (rank + 1)} T5\n')

print('Done!')
