import json
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--results", type=str, required=True)
parser.add_argument("--t5_output_ids", type=str, required=True)
parser.add_argument("--t5_output", type=str, required=True)
parser.add_argument("--topk", type=int, default=3)

args = parser.parse_args()

abstract_rerank_ids = open(args.t5_output_ids, "r").readlines()
abstract_rerank = open(args.t5_output, "r").readlines()
abstract_retrieval = open(args.results, "w")

claim_abstract_score_dict = {}
for idx in tqdm(range(len(abstract_rerank_ids))):
    claim_id, abstract_id, _ = abstract_rerank_ids[idx].split()
    _, score = abstract_rerank[idx].split()
    score = float(score)
    if claim_id not in claim_abstract_score_dict:
        claim_abstract_score_dict[claim_id] = {}
    if abstract_id not in claim_abstract_score_dict[claim_id]:
        claim_abstract_score_dict[claim_id][abstract_id] = score
    elif score > claim_abstract_score_dict[claim_id][abstract_id]:
        claim_abstract_score_dict[claim_id][abstract_id] = score

for claim_id in claim_abstract_score_dict:
    sorted_abstract_order = sorted(claim_abstract_score_dict[claim_id].items(), key=lambda x: x[1], reverse=True)
    selected_abstracts = [int(tup[0]) for tup in sorted_abstract_order[:args.topk]]
    abstract_retrieval.write(json.dumps({"claim_id": int(claim_id), "doc_ids": selected_abstracts}) + "\n")
