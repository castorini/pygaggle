import argparse
import ftfy
import json
import os
import random

from fever_utils import extract_sentences, make_sentence_id, normalize_text, split_sentence_id, truncate

def generate_data(args):
    queries = {}
    labels = {}
    evidences = {}
    docs = {}

    num_truncated = 0

    # read in dataset file and save queries and evidences to dicts
    with open(args.dataset_file, 'r', encoding='utf-8') as f:
        print('Reading FEVER dataset file...')
        for line in f:
            line_json = json.loads(line.strip())
            query_id = line_json['id']

            query = line_json['claim']
            queries[query_id] = query

            label = line_json['label']
            if label == 'SUPPORTS':
                labels[query_id] = 'true'
            elif label == 'REFUTES':
                labels[query_id] = 'false'
            else:  # label == 'NOT ENOUGH INFO'
                labels[query_id] = 'weak'

            annotators = []
            if label != 'NOT ENOUGH INFO':  # no evidence set for NEI queries, will sample from run files later
                for annotator in line_json['evidence']:
                    evidence_set = []
                    for evidence in annotator:
                        evidence[2] = ftfy.fix_text(evidence[2])
                        docs[evidence[2]] = 'N/A'  # placeholder
                        evidence_set.append(make_sentence_id(evidence[2], evidence[3]))
                    annotators.append(evidence_set)
            evidences[query_id] = annotators

    # samples evidence from pred_sent_ids
    def negative_sample(query_id, pred_sent_ids):
        neg_sent_ids = random.sample(pred_sent_ids, random.randint(1, args.max_evidences))

        for sent_id in neg_sent_ids:
            doc_id, _ = split_sentence_id(sent_id)
            docs[doc_id] = 'N/A'  # placeholder

        return [neg_sent_ids]

    # read in run file and sample run file ranking predictions for queries
    with open(args.run_file, 'r', encoding='utf-8') as f:
        print('Reading run file...')
        curr_query = None
        pred_sent_ids = []
        for line in f:
            query_id, sent_id, rank = line.strip().split('\t')
            query_id = int(query_id)

            # if we reach a new query in the run file, perform sampling for previous query if needed
            if query_id != curr_query:
                if curr_query is not None and len(evidences[curr_query]) == 0:
                    evidences[curr_query] = negative_sample(curr_query, pred_sent_ids)
                curr_query = query_id
                pred_sent_ids.clear()

            if args.min_rank <= int(rank) <= args.max_rank:
                pred_sent_ids.append(sent_id)

        # handle the final query
        if len(evidences[curr_query]) == 0:
            evidences[curr_query] = negative_sample(curr_query, pred_sent_ids)

    # read through all wiki dump files and save doc text for involved docs
    print('Reading wiki pages...')
    for file in os.listdir(args.collection_folder):
        with open(os.path.join(args.collection_folder, file), 'r', encoding='utf-8') as f:
            for line in f:
                line_json = json.loads(line.strip())
                if line_json['id'] in docs:
                    docs[line_json['id']] = line_json['lines']

    # write query-doc text pairs to files
    with open(args.output_id_file, 'w', encoding='utf-8') as f_id, \
            open(args.output_text_file, 'w', encoding='utf-8') as f_text:
        print('Writing query-doc pairs to files...')
        for query_id, query_text in queries.items():
            label = labels[query_id]

            for evidence_ids in evidences[query_id]:
                evidence_texts = []
                for evidence in evidence_ids:
                    # get specific sentence from within doc_text
                    doc_id, sent_num = split_sentence_id(evidence)
                    entity = doc_id.replace('_', ' ')  # prepend entity name to document text
                    doc_text = docs[doc_id]
                    sent_text, _ = extract_sentences(doc_text)[sent_num]
                    evidence_texts.append(f'{normalize_text(entity)} . {normalize_text(sent_text)}')

                if args.format == 'concat':
                    evidence_ids_str = ' '.join(evidence_ids)
                    prefixed_evidence_texts = []
                    for i, evidence_text in enumerate(evidence_texts):
                        truncated_text, num_truncated = truncate(query_text, evidence_text, args.max_evidences,
                                                                 args.max_seq_len, num_truncated)
                        prefixed_evidence_texts.append(f'sentence{i + 1}: {truncated_text}')
                    evidence_texts_str = ' '.join(prefixed_evidence_texts)

                    f_id.write(f'{query_id}\t{evidence_ids_str}\n')
                    f_text.write(f'hypothesis: {query_text} {evidence_texts_str}\t{label}\n')
                else:  # args.format == 'agg'
                    for evidence_id, evidence_text in zip(evidence_ids, evidence_texts):
                        f_id.write(f'{query_id}\t{evidence_id}\n')
                        f_text.write(f'hypothesis: {query_text} premise: {evidence_text}\t{label}\n')

    print(f'Number of sentences truncated: {num_truncated}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates "gold" FEVER label prediction training data.')
    parser.add_argument('--dataset_file', required=True, help='FEVER dataset file.')
    parser.add_argument('--run_file', required=True, help='Run file generated after re-ranking.')
    parser.add_argument('--collection_folder', required=True, help='FEVER wiki-pages directory.')
    parser.add_argument('--output_id_file', required=True, help='Output query-doc id pairs file.')
    parser.add_argument('--output_text_file', required=True, help='Output query-doc text pairs file.')
    parser.add_argument('--min_rank', type=int, help='Smallest rank to sample from.')
    parser.add_argument('--max_rank', type=int, help='Largest rank to sample from.')
    parser.add_argument('--max_evidences', type=int, required=True, help='Max number of evidences to negative sample.')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Max number of tokens per line.')
    parser.add_argument('--format', required=True, choices=['concat', 'agg'], help='Format of output query-doc files.')
    parser.add_argument('--seed', type=int, help='Optional seed for random sampling.')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    generate_data(args)

    print('Done!')
