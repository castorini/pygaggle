import jsonlines
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=str, default="../scifact/data/corpus.jsonl")
parser.add_argument("--claims", type=str, default="../scifact/data/claims_train.jsonl")
parser.add_argument("--t5_input", type=str, required=True)
parser.add_argument("--title", action="store_true")
parser.add_argument("--balanced", action='store_true')
args = parser.parse_args()

true_counter = 0
false_counter = 0
t5_input = open(args.t5_input, "w")
corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
for claim in tqdm(jsonlines.open(args.claims)):
    for doc_id, evidence in claim['evidence'].items():
        doc = corpus[int(doc_id)]
        evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
        title = doc['title'][:-1] if doc['title'][-1] == '.' else doc['title']
        for i, sentence in enumerate(doc['abstract']):
            qtext = claim['claim']
            dtext = sentence.rstrip().replace('\n', ' ')
            evidence = str(i in evidence_sentence_idx).lower()
            if evidence == "true":
                true_counter += 1
            else:
                false_counter += 1
            if args.title:
                if args.balanced:
                    if evidence == "true":
                        t5_input.write(f'Query: {qtext} Document: {title}. {dtext} Relevant:\t{evidence}\n')
                        t5_input.write(f'Query: {qtext} Document: {title}. {dtext} Relevant:\t{evidence}\n')
                        t5_input.write(f'Query: {qtext} Document: {title}. {dtext} Relevant:\t{evidence}\n')

                t5_input.write(f'Query: {qtext} Document: {title}. {dtext} Relevant:\t{evidence}\n')
            else:
                if args.balanced:
                    if evidence == "true":
                        t5_input.write(f'Query: {qtext} Document: {dtext} Relevant:\t{evidence}\n')
                        t5_input.write(f'Query: {qtext} Document: {dtext} Relevant:\t{evidence}\n')
                        t5_input.write(f'Query: {qtext} Document: {dtext} Relevant:\t{evidence}\n')
                t5_input.write(f'Query: {qtext} Document: {dtext} Relevant:\t{evidence}\n')
print(f'True: {true_counter}')
print(f'False: {false_counter}')
t5_input.close()
