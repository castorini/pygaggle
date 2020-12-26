import jsonlines
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=str, default="data/corpus.jsonl")
parser.add_argument("--claims", type=str, required=True)
parser.add_argument("--retrieval", type=str, required=True)
parser.add_argument("--t5_input_ids", type=str, required=True)
parser.add_argument("--t5_input", type=str, required=True)
parser.add_argument("--stride", type=int, default=4)
parser.add_argument("--length", type=int, default=8)
args = parser.parse_args()

abstracts = {doc['doc_id']: doc["abstract"] for doc in jsonlines.open(args.corpus)}
claims = {doc['id']: doc["claim"] for doc in jsonlines.open(args.claims)}
retrieval = {doc['claim_id']: doc["doc_ids"] for doc in jsonlines.open(args.retrieval)}
abstract_rerank_ids = open(args.t5_input_ids, "w")
abstract_rerank = open(args.t5_input, "w")

for claim_id in tqdm(retrieval):
    for doc_id in retrieval[claim_id]:
        sentences = [sent.strip() for sent in abstracts[doc_id]]
        idx = 0
        segment = []
        while idx < len(sentences):
            segment.append(sentences[idx].strip())
            if idx == len(sentences) - 1 or len(segment) == args.length:
                claim = claims[claim_id]
                document = " ".join(segment)
                start_idx = idx - len(segment) + 1
                abstract_rerank_ids.write(f"{claim_id}\t{doc_id}\t{start_idx}\n")
                to_write = f"Query: {claim} Document: {document} Relevant:\n"
                abstract_rerank.write(to_write)

            if idx != len(sentences) - 1 and len(segment) == args.length:
                segment = []
                idx = idx - args.stride
            idx += 1

abstract_rerank_ids.close()
abstract_rerank.close()
