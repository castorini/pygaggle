import jsonlines
import argparse
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", type=str, default="../scifact/data/corpus.jsonl")
parser.add_argument("--claims", type=str, default="../scifact/data/claims_train.jsonl")
parser.add_argument("--t5_input", type=str, required=True)
parser.add_argument("--no-weak", action="store_true")
args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
label_encodings = {'CONTRADICT': 'false', 'NOT_ENOUGH_INFO': 'weak', 'SUPPORT': 'true'}
t5_input = open(args.t5_input, 'w')
pos_counter = 0
neg_counter = 0
weak_counter = 0
for claim in tqdm(jsonlines.open(args.claims)):
    if claim['evidence']:
        for doc_id, evidence_sets in claim['evidence'].items():
            doc = corpus[int(doc_id)]

            # Add individual evidence set as samples:
            # for evidence_set in evidence_sets:
            #     hypothesis = claim['claim']
            #    rationale = [doc['abstract'][i].strip() for i in evidence_set['sentences']]
            #    label = label_encodings[evidence_set['label']]
            #    evidence = ' '.join(["Evidence{}: ".format(idx + 1) + sent.replace("\n", ' ').strip() for idx, sent in enumerate(rationale)])
            #    t5_input.write(f"Hypothesis: {hypothesis} {evidence} Judgement:\t{label}\n")
            #    if label == "true":
            #        pos_counter += 1
            #    elif label == 'false':
            #        neg_counter += 1
            #    elif label == "weak":
            #        weak_counter += 1

            # Add all evidence sets as positive samples
            rationale_idx = {s for es in evidence_sets for s in es['sentences']}
            rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(rationale_idx))]
            label = label_encodings[evidence_sets[0]['label']]
            hypothesis = claim['claim']
            evidence = ' '.join(["sentence{}: ".format(idx + 1) + sent.replace("\n", ' ').strip() for idx, sent in enumerate(rationale_sentences)])
            t5_input.write(f"hypothesis: {hypothesis} {evidence}\t{label}\n")
            if label == "true":
                pos_counter += 1
                t5_input.write(f"hypothesis: {hypothesis} {evidence}\t{label}\n")
                t5_input.write(f"hypothesis: {hypothesis} {evidence}\t{label}\n")
            elif label == 'false':
                neg_counter += 1
                t5_input.write(f"hypothesis: {hypothesis} {evidence}\t{label}\n")
                t5_input.write(f"hypothesis: {hypothesis} {evidence}\t{label}\n")
                t5_input.write(f"hypothesis: {hypothesis} {evidence}\t{label}\n")
                t5_input.write(f"hypothesis: {hypothesis} {evidence}\t{label}\n")
            elif label == "weak":
                weak_counter += 1

            # Add negative samples
            non_rationale_idx = set(range(len(doc['abstract']))) - rationale_idx
            non_rationale_idx = random.sample(non_rationale_idx,
                                              k=min(random.randint(1, 2), len(non_rationale_idx)))
            non_rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(non_rationale_idx))]
            label = label_encodings['NOT_ENOUGH_INFO']
            hypothesis = claim['claim']
            evidence = ' '.join(["sentence{}: ".format(idx + 1) + sent.replace("\n", ' ').strip() for idx, sent in
                                 enumerate(non_rationale_sentences)])
            if not args.no_weak:
                t5_input.write(f"hypothesis: {hypothesis} {evidence}\t{label}\n")
            if label == "true":
                pos_counter += 1
            elif label == 'false':
                neg_counter += 1
            elif label == "weak":
                weak_counter += 1
    else:
        # Add negative samples
        for doc_id in claim['cited_doc_ids']:
            doc = corpus[int(doc_id)]
            non_rationale_idx = random.sample(range(len(doc['abstract'])), k=random.randint(1, 2))
            non_rationale_sentences = [doc['abstract'][i].strip() for i in non_rationale_idx]
            label = label_encodings['NOT_ENOUGH_INFO']
            hypothesis = claim['claim']
            evidence = ' '.join(["sentence{}: ".format(idx + 1) + sent.replace("\n", ' ').strip() for idx, sent in
                                 enumerate(non_rationale_sentences)])
            if not args.no_weak:
                t5_input.write(f"hypothesis: {hypothesis} {evidence}\t{label}\n")
            if label == "true":
                pos_counter += 1
            elif label == 'false':
                neg_counter += 1
            elif label == "weak":
                weak_counter += 1
t5_input.close()
print(f"Pos: {pos_counter}")
print(f"Neg: {neg_counter}")
print(f"Weak: {weak_counter}")
