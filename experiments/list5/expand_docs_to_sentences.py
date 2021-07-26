import argparse
import json
import os

from fever_utils import extract_sentences, make_sentence_id

def convert_run(args):
    doc_sentences = {}
    rankings = {}

    # read in input run file and save rankings to dict
    with open(args.input_run_file, 'r', encoding='utf-8') as f:
        print('Reading run file...')
        for line in f:
            query_id, doc_id, rank = line.strip().split('\t')
            if doc_id not in doc_sentences:
                doc_sentences[doc_id] = []
            if query_id not in rankings:
                rankings[query_id] = []
            rankings[query_id].append(doc_id)

    # read through all wiki dump files and save sentence IDs for involved docs
    print('Reading wiki pages...')
    for file in os.listdir(args.collection_folder):
        with open(os.path.join(args.collection_folder, file), 'r', encoding='utf-8') as f:
            for line in f:
                line_json = json.loads(line.strip())
                if line_json['id'] in doc_sentences:
                    sent_ids = [id for sent, id in extract_sentences(line_json['lines']) if sent]
                    doc_sentences[line_json['id']].extend(sent_ids)

    # write expanded sentence IDs to output run file
    with open(args.output_run_file, 'w', encoding='utf-8') as f:
        print('Writing sentences to run file...')
        for query_id, doc_ids in rankings.items():
            query_index = 1
            for doc_id in doc_ids[:args.k]:
                for sent_num in doc_sentences[doc_id]:
                    sent_id = make_sentence_id(doc_id, sent_num)
                    f.write(f'{query_id}\t{sent_id}\t{query_index}\n')
                    query_index += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Expands document-level anserini run file to sentence-level.')
    parser.add_argument('--input_run_file', required=True, help='Input document-level run file.')
    parser.add_argument('--collection_folder', required=True, help='FEVER wiki-pages directory.')
    parser.add_argument('--output_run_file', required=True, help='Output sentence-level run file.')
    parser.add_argument('--k', default=100, type=int, help='Top k documents to expand.')
    args = parser.parse_args()

    convert_run(args)

    print('Done!')
