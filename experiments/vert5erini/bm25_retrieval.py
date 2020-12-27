import json
from pyserini.search import SimpleSearcher
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=str, default="index")
parser.add_argument("--claims", type=str, required=True)
parser.add_argument("--results", type=str, required=True)
parser.add_argument("--topk", type=int, default=20)
args = parser.parse_args()

searcher = SimpleSearcher(args.index)
claim_f = open(args.claims, "r").readlines()
retrieval_f = open(args.results, "w")
for line in tqdm(claim_f):
    info = json.loads(line)
    claim = info["claim"]
    claim_id = info["id"]
    hits = searcher.search(claim, args.topk)
    doc_ids = []
    for hit in hits:
        doc_ids.append(int(hit.docid))
    retrieval_f.write(json.dumps({"claim_id": claim_id, "doc_ids": doc_ids})+"\n")
retrieval_f.close()
