import argparse
import ftfy
import itertools
import json
import os

from fever_utils import extract_entities, extract_sentences, make_sentence_id, normalize_text, split_sentence_id

def convert_run(args):
    queries = {}
    evidences = {}
    pred_evidences = {}
    docs = {}

    # read in dataset file and save queries and evidences to dicts
    with open(args.dataset_file, 'r', encoding='utf-8') as f:
        print('Reading FEVER dataset file...')
        for line in f:
            line_json = json.loads(line.strip())

            query_id = line_json['id']

            query = line_json['claim']
            queries[query_id] = query

            # only save evidences for non-test sets and non-NEI queries
            deduped_evidence_set = set()
            if args.has_labels and line_json['label'] != 'NOT ENOUGH INFO':
                for annotator in line_json['evidence']:
                    for evidence in annotator:
                        evidence[2] = ftfy.fix_text(evidence[2])
                        docs[evidence[2]] = 'N/A'  # placeholder
                        deduped_evidence_set.add(make_sentence_id(evidence[2], evidence[3]))
            evidences[query_id] = deduped_evidence_set

    # read in run file and save rankings to dict
    with open(args.run_file, 'r', encoding='utf-8') as f:
        print('Reading run file...')
        for line in f:
            query_id, sent_id, rank = line.strip().split('\t')
            query_id = int(query_id)
            doc_id, _ = split_sentence_id(sent_id)
            docs[doc_id] = 'N/A'  # placeholder
            if query_id not in pred_evidences:
                pred_evidences[query_id] = []
            if args.k is None or int(rank) <= args.k:
                pred_evidences[query_id].append(sent_id)

    # read through all wiki dump files and save doc text for involved docs
    print('Reading wiki pages...')
    for file in os.listdir(args.collection_folder):
        with open(os.path.join(args.collection_folder, file), 'r', encoding='utf-8') as f:
            for line in f:
                line_json = json.loads(line.strip())
                if line_json['id'] in docs:
                    docs[line_json['id']] = line_json['lines']

    # write query-doc pairs to files
    with open(args.output_id_file, 'w', encoding='utf-8') as f_id, \
            open(args.output_text_file, 'w', encoding='utf-8') as f_text:
        print('Writing query-doc pairs to files...')
        for query_id, sent_ids in pred_evidences.items():
            query_text = queries[query_id]
            if args.type == 'mono':
                if args.ner:
                    ner_entities = extract_entities(query_text)

                for rank, sent_id in enumerate(sent_ids):
                    if args.has_labels:
                        relevance = 'true' if sent_id in evidences[query_id] else 'false'

                    # get specific sentence from within doc_text
                    doc_id, sent_num = split_sentence_id(sent_id)
                    entity = doc_id.replace('_', ' ')  # prepend entity name to document text
                    doc_text = docs[doc_id]
                    sent_text, _ = extract_sentences(doc_text)[sent_num]

                    # write query-doc pair ids and texts
                    if args.has_labels:
                        f_id.write(f'{query_id}\t{sent_id}\t{rank + 1}\t{relevance}\n')
                    else:
                        f_id.write(f'{query_id}\t{sent_id}\t{rank + 1}\n')
                    if args.ner:
                        numbered_entities = [f'Entity{i + 1}: {entity}' for i, entity in enumerate(ner_entities)]
                        entities_str = ' '.join(numbered_entities)
                        f_text.write(
                            f'Query: {query_text} Document: {entity} . {normalize_text(sent_text)} {entities_str} Relevant:\n'
                        )
                    else:
                        f_text.write(
                            f'Query: {query_text} Document: {entity} . {normalize_text(sent_text)} Relevant:\n')
            else:  # args.type == 'duo'
                ranked_sent_ids = [(sent_id, i) for i, sent_id in enumerate(sent_ids)]
                for (sent_id_1, rank_1), (sent_id_2, rank_2) in itertools.permutations(ranked_sent_ids, 2):
                    if args.has_labels:
                        relevance = 'true' if sent_id_1 in evidences[query_id] else 'false'

                    # get specific sentence from within doc_text
                    doc_id_1, sent_1_num = split_sentence_id(sent_id_1)
                    entity_1 = doc_id_1.replace('_', ' ')  # prepend entity name to document text
                    doc_text_1 = docs[doc_id_1]
                    sent_1_text, _ = extract_sentences(doc_text_1)[sent_1_num]

                    doc_id_2, sent_2_num = split_sentence_id(sent_id_2)
                    entity_2 = doc_id_2.replace('_', ' ')  # prepend entity name to document text
                    doc_text_2 = docs[doc_id_2]
                    sent_2_text, _ = extract_sentences(doc_text_2)[sent_2_num]

                    # write query-doc pair ids and texts
                    if args.has_labels:
                        f_id.write(f'{query_id}\t{sent_id_1}\t{rank_1 + 1}\t{sent_id_2}\t{rank_2 + 1}\t{relevance}\n')
                    else:
                        f_id.write(f'{query_id}\t{sent_id_1}\t{rank_1 + 1}\t{sent_id_2}\t{rank_2 + 1}\n')
                    f_text.write(
                        f'Query: {query_text} Document1: {entity_1} . {normalize_text(sent_1_text)} Document2: {entity_2} . {normalize_text(sent_2_text)} Relevant:\n'
                    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts run files to T5 sentence re-ranking model input format.')
    parser.add_argument('--dataset_file', required=True, help='FEVER dataset file.')
    parser.add_argument('--run_file', required=True, help='Run file from running retrieval with anserini.')
    parser.add_argument('--collection_folder', required=True, help='FEVER wiki-pages directory.')
    parser.add_argument('--output_id_file', required=True, help='Output query-doc id pairs file.')
    parser.add_argument('--output_text_file', required=True, help='Output query-doc text pairs file.')
    parser.add_argument('--k', type=int, help='Number of top sentences to include for re-ranking.')
    parser.add_argument('--type', required=True, choices=['mono', 'duo'], help='Type of T5 inference.')
    parser.add_argument('--has_labels', action='store_true', help='Whether the dataset file is labelled.')
    parser.add_argument('--ner', action='store_true', help='Whether to append NER entities (only for mono re-ranking).')
    args = parser.parse_args()

    convert_run(args)

    print('Done!')
