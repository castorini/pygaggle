import collections
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--t5_output", type=str, required=True)
parser.add_argument("--t5_output_ids", type=str, required=True)
parser.add_argument("--msmarco_run", type=str, required=True)
args = parser.parse_args()

examples = collections.defaultdict(list)
with open(args.query_run_ids) as f_gt, open(args.predictions_path) as f_pred:
    for line_gt, line_pred in zip(f_gt, f_pred):
        query_id, doc_id = line_gt.strip().split('\t')
        _, score = line_pred.strip().split('\t')
        score = float(score)
        examples[query_id].append((doc_id, score))

with open(args.output_path, 'w') as fout:
    for query_id, doc_ids_scores in examples.items():
        doc_ids_scores.sort(key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(doc_ids_scores):
            fout.write(f'{query_id}\t{doc_id}\t{rank + 1}\n')
