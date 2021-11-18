"""Script to convert T5 predictions into a TREC-formatted run."""
import argparse
import collections
import numpy as np
import xml.etree.ElementTree as ET

def parse_xml(input_file, year="2020"):
    xml = ET.parse(input_file)
    root = xml.getroot()
    query = {}
    for child in root:
        qid = child.find('number').text
        if year=="2020":
            query[qid] = True if child.find('answer').text == "yes" else False
        else:
            query[qid] = True if child.find('stance').text == "helpful" else False
    return query

parser = argparse.ArgumentParser(
    description='Convert T5 predictions into a TREC-formatted run.')
parser.add_argument('--predictions', type=str, required=True, help='T5 predictions file.')
parser.add_argument('--query_run_ids', type=str, required=True,
                    help='File containing query doc id pairs paired with the T5\'s predictions file.')
parser.add_argument('--output', type=str, required=True, help='run file in the TREC format.')
parser.add_argument('--year', type=str, required=True, help='run file in the TREC format.')
parser.add_argument("--input_topic", required=True, help='input path to trec xml topic files')


args = parser.parse_args()

labels = parse_xml(args.input_topic, args.year)

examples = collections.defaultdict(lambda: collections.defaultdict(list))
with open(args.query_run_ids) as f_query_run_ids, open(args.predictions) as f_pred:
    for line_query_doc_id, line_pred in zip(f_query_run_ids, f_pred):
        query_id, doc_id = line_query_doc_id.strip().split()
        _, score_t, score_f, score_w = line_pred.strip().split()

        abstract_id, seg_id = doc_id.split("#")
        score = np.exp(float(score_t))  - np.exp(float(score_f))
        if labels[query_id] == False:
            score = 0 - score
        examples[query_id][abstract_id].append((score, doc_id))

with open(args.output, 'w') as fout:
    for query_id, doc_ids_scores in examples.items():
        doc_ids_scores = [(doc_id, max(scores, key=lambda x: x[0])) for doc_id, scores in doc_ids_scores.items()]
        doc_ids_scores.sort(key=lambda x: x[1][0], reverse=True)
        for rank, (doc_id, (score, segid)) in enumerate(doc_ids_scores):
            fout.write(f'{query_id} Q0 {doc_id} {rank + 1} {score} monot5\n')

print('Done!')
