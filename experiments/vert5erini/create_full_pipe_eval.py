import jsonlines

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--ar_result", type=str, required=True)
parser.add_argument("--ss_result", type=str, required=True)
parser.add_argument("--lp_result", type=str, required=True)
# parser.add_argument("--ss_eval", type=str, required=True)
parser.add_argument("--lp_eval", type=str, required=True)
args = parser.parse_args()


claim_labels = jsonlines.open(args.lp_result)
evaluate_file = jsonlines.open(args.lp_eval, mode="w")
abstract_retrieval = jsonlines.open(args.ar_result)
rationale_selection = jsonlines.open(args.ss_result)
# rationale_selection_eval = jsonlines.open(args.ss_eval, mode="w")

selection_dict = {}
for line in rationale_selection:
    claim_id = line["claim_id"]
    for abstract in line["evidence"]:
        sentences = line["evidence"][abstract]
        selection_dict[f"{claim_id}#{abstract}"] = sentences

label_dict = {}
for line in claim_labels:
    claim_id = line["claim_id"]
    for abstract in line["labels"]:
        label = line["labels"][abstract]["label"]
        label_dict[f"{claim_id}#{abstract}"] = label
for line in abstract_retrieval:
    claim_id = line["claim_id"]
    labels = {}
    for abstract in line["doc_ids"]:
        key = f"{claim_id}#{abstract}"
        if key in label_dict:
            label = label_dict[key]
        else:
            label = "NOT_ENOUGH_INFO"
        if key in selection_dict:
            sentences = selection_dict[key]
        else:
            sentences = []
        if label != "NOT_ENOUGH_INFO":
            labels[str(abstract)] = {"label": label, "sentences": sentences, "confidence": 1}
    evaluate_file.write({"id": claim_id, "evidence": labels})
