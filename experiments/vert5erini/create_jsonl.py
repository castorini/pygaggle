import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

output = []
with open(args.corpus) as f:
    for line in f:
        info = json.loads(line)
        docid = info["docid"]
        title = info["title"]
        abstract = info["abstract"]
        abstract = " ".join([sent.strip() for sent in abstract]).replace("\n", " ")
        output.append({
            "id": str(docid),
            "contents": f"{title}\n{abstract}"
        })
json.dump(output, open(args.output, "w"))
