import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=str, required=True)
parser.add_argument("--claims", type=str, required=True)
parser.add_argument("--t5_output_ids", type=str, required=True)
parser.add_argument("--t5_output", type=str, required=True)
args = parser.parse_args()

label_map = {"weak": "NOT_ENOUGH_INFO", "false": "CONTRADICT", "true": "SUPPORT"}

format1_ids = open(args.t5_output_ids, "r")
format1_label = open(args.t5_output, "r")
format1_eval = open(args.results, "w")
labels = [line.split()[0] for line in format1_label.readlines()]
id_lines = format1_ids.readlines()
claim_labels_dict = {}
for idx, line in enumerate(id_lines):
    info = line.split()
    if info[0] not in claim_labels_dict:
        claim_labels_dict[str(info[0])] = {}
    claim_labels_dict[info[0]][info[1]] = {"label": label_map[labels[idx]]}

claims_f = open(args.claims, "r").readlines()
all_ids = []
for line in claims_f:
    info = json.loads(line)
    claim_id = info["id"]
    all_ids.append(claim_id)

for key in all_ids:
    if str(key) in claim_labels_dict:
        format1_eval.write(json.dumps({"claim_id": int(key), "labels": claim_labels_dict[str(key)]})+"\n")
    else:
        format1_eval.write(json.dumps({"claim_id": int(key), "labels": {}}) + "\n")
format1_eval.close()
format1_label.close()
