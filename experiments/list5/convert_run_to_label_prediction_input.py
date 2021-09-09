import argparse
import json
import os

from fever_utils import extract_sentences, make_sentence_id, normalize_text, split_sentence_id, truncate

def convert_run(args):
    queries = {}
    labels = {}
    evidences = {}
    docs = {}

    num_truncated = 0

    # read in dataset file and save queries to dicts
    with open(args.dataset_file, 'r', encoding='utf-8') as f:
        print('Reading FEVER dataset file...')
        for line in f:
            line_json = json.loads(line.strip())
            query_id = line_json['id']

            query = line_json['claim']
            queries[query_id] = query

            if args.has_labels:
                label = line_json['label']
                if label == 'SUPPORTS':
                    labels[query_id] = 'true'
                elif label == 'REFUTES':
                    labels[query_id] = 'false'
                else:  # label == 'NOT ENOUGH INFO'
                    labels[query_id] = 'weak'

    def generate_samples(query_id, pred_sent_ids):
        evidence_sets = []
        if args.format == 'concat':
            evidence_sets = [[sent_id for sent_id in pred_sent_ids]]
        elif args.format == 'agg':
            evidence_sets = [[sent_id] for sent_id in pred_sent_ids]
        else:  # args.format == 'seq':
            curr_preds = []
            for sent_id in pred_sent_ids:
                curr_preds.append(sent_id)
                evidence_sets.append([pred for pred in curr_preds])

        return evidence_sets

    # read in run file and take top run file ranking predictions
    with open(args.run_file, 'r', encoding='utf-8') as f:
        print('Reading run file...')
        curr_query = None
        pred_sent_ids = []
        for line in f:
            query_id, sent_id, rank = line.strip().split('\t')
            query_id = int(query_id)

            # if we reach a new query in the run file, generate samples for previous query
            if query_id != curr_query:
                if curr_query is not None:
                    evidences[curr_query] = generate_samples(curr_query, pred_sent_ids)
                curr_query = query_id
                pred_sent_ids.clear()

            if int(rank) <= args.max_evidences:
                doc_id, _ = split_sentence_id(sent_id)
                docs[doc_id] = 'N/A'  # placeholder
                pred_sent_ids.append(sent_id)

        # handle the final query
        evidences[curr_query] = generate_samples(curr_query, pred_sent_ids)

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
            if args.has_labels:
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

                # format evidence ids and texts in proper format
                evidence_ids_str = ' '.join(evidence_ids)
                prefixed_evidence_texts = []
                for i, evidence_text in enumerate(evidence_texts):
                    if args.format == 'agg':
                        prefixed_evidence_texts.append(f'premise: {evidence_text}')
                    else:
                        truncated_text, num_truncated = truncate(query_text, evidence_text, args.max_evidences,
                                                                 args.max_seq_len, num_truncated)
                        prefixed_evidence_texts.append(f'sentence{i + 1}: {truncated_text}')
                evidence_texts_str = ' '.join(prefixed_evidence_texts)

                if args.has_labels:
                    f_id.write(f'{query_id}\t{evidence_ids_str}\t{label}\n')
                else:
                    f_id.write(f'{query_id}\t{evidence_ids_str}\n')
                f_text.write(f'hypothesis: {query_text} {evidence_texts_str}\n')

    print(f'Number of sentences truncated: {num_truncated}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts run files to T5 label prediction model input format.')
    parser.add_argument('--dataset_file', required=True, help='FEVER dataset file.')
    parser.add_argument('--run_file', required=True, help='Run file generated after re-ranking.')
    parser.add_argument('--collection_folder', required=True, help='FEVER wiki-pages directory.')
    parser.add_argument('--output_id_file', required=True, help='Output query-doc id pairs file (empty for test set).')
    parser.add_argument('--output_text_file', required=True, help='Output query-doc text pairs file.')
    parser.add_argument('--max_evidences', type=int, default=5, help='Max concatenated evidences per line.')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Max number of tokens per line.')
    parser.add_argument('--format', required=True, choices=['concat', 'agg', 'seq'])
    parser.add_argument('--has_labels', action='store_true', help='Whether the dataset file is labelled.')
    args = parser.parse_args()

    convert_run(args)

    print('Done!')
