import jsonlines
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=str, default="data/corpus.jsonl")
parser.add_argument("--claims", type=str, required=True)
parser.add_argument("--sentence_selection", type=str, required=True)
parser.add_argument("--t5_input_ids", type=str, required=True)
parser.add_argument("--t5_input", type=str, required=True)
args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
dataset = jsonlines.open(args.claims)
rationale_selection = jsonlines.open(args.sentence_selection)
label_encodings = {'CONTRADICT': "false", 'NOT_ENOUGH_INFO': "weak", 'SUPPORT': "true"}

format3_dev_ids = open(args.t5_input_ids, "w")
format3_dev = open(args.t5_input, "w")
for data, selection in tqdm(list(zip(dataset, rationale_selection))):
    assert data['id'] == selection['claim_id']
    claim_id = data['id']
    claim = data['claim']
    for doc_id, indices in selection['evidence'].items():
        evidence = ' '.join(["sentence{}: ".format(idx+1) + corpus[int(doc_id)]['abstract'][i].strip() for idx, i in enumerate(indices)])
        format3_dev_ids.write("{}\t{}\n".format(claim_id, doc_id))
        format3_dev.write("hypothesis: {} {}:\n".format(claim, evidence))
format3_dev_ids.close()
format3_dev.close()