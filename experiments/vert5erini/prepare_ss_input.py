import json
import jsonlines
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=str, default="data/corpus.jsonl")
parser.add_argument("--claims", type=str, required=True)
parser.add_argument("--retrieval", type=str, required=True)
parser.add_argument("--t5_input_ids", type=str, required=True)
parser.add_argument("--t5_input", type=str, required=True)
parser.add_argument("--title", action="store_true")
args = parser.parse_args()

abstracts = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
abstract_retrieval = {doc["claim_id"]: doc["doc_ids"] for doc in jsonlines.open(args.retrieval)}
claim_f = open(args.claims, "r")
t5_input = open(args.t5_input, "w")
t5_input_ids = open(args.t5_input_ids, "w")
for line in tqdm(claim_f):
    claim_info = json.loads(line)
    for abstract_id in abstract_retrieval[claim_info["id"]]:
        sentences = abstracts[abstract_id]["abstract"]
        doc = abstracts[abstract_id]
        title = doc['title'][:-1] if doc['title'][-1] == '.' else doc['title']
        for idx, sent in enumerate(sentences):
            qtext = claim_info["claim"]
            dtext = sent.strip()
            if args.title:
                t5_input.write(f'Query: {qtext} Document: {title}. {dtext} Relevant:\n')
            else:
                t5_input.write(f'Query: {qtext} Document: {dtext} Relevant:\n')
            qid = claim_info["id"]
            did = str(str(abstract_id)) + "#{}".format(idx)
            t5_input_ids.write(f'{qid}\t{did}\n')
t5_input.close()
claim_f.close()
t5_input_ids.close()
