import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=str, required=True)
parser.add_argument("--claims", type=str, required=True)
parser.add_argument("--t5_output_ids", type=str, required=True)
parser.add_argument("--t5_output", type=str, required=True)
parser.add_argument("--thres", type=float, required=True)
args = parser.parse_args()

claim_sent_ids_f = open(args.t5_output_ids, "r")
rationale_selection = open(args.t5_output, "r")
ids = claim_sent_ids_f.readlines()
scores = rationale_selection.readlines()
rationale_selection.close()
claim_sent_ids_f.close()
result = {}
for i in tqdm(range(len(ids))):
    sent_info = ids[i].split()
    claim_id = int(sent_info[0])
    abstract_id, sent_idx = sent_info[1].split("#")
    abstract_id = int(abstract_id)
    sent_idx = int(sent_idx)
    label = float(scores[i].split()[1]) > -1*args.thres
    if str(claim_id) not in result:
        result[str(claim_id)] = {}
    if label:
        if str(abstract_id) not in result[str(claim_id)]:
            result[str(claim_id)][str(abstract_id)] = [sent_idx]
        else:
            result[str(claim_id)][str(abstract_id)].append(sent_idx)
claims_f = open(args.claims, "r").readlines()
all_ids = []
for line in claims_f:
    info = json.loads(line)
    claim_id = info["id"]
    all_ids.append(claim_id)

rationale_selection_jsonl = open(args.results, "w")
for key in all_ids:
    if str(key) in result:
        rationale_selection_jsonl.write(json.dumps({"claim_id": int(key), "evidence": result[str(key)]})+"\n")
    else:
        rationale_selection_jsonl.write(json.dumps({"claim_id": int(key), "evidence": {}}) + "\n")
rationale_selection_jsonl.close()
