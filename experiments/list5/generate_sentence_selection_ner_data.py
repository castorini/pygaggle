import argparse
import ftfy
import json
import os
import random

from fever_utils import extract_entities, extract_sentences, make_sentence_id, normalize_text, remove_disambiguation, split_sentence_id

def generate_data(args):
    queries = {}
    evidences = {}
    pred_evidences = {}
    docs = {}

    num_actual = 0
    num_pred = 0
    correct = 0

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
            if line_json['label'] != 'NOT ENOUGH INFO':
                for annotator in line_json['evidence']:
                    for evidence in annotator:
                        evidence[2] = ftfy.fix_text(evidence[2])
                        docs[evidence[2]] = 'N/A'  # placeholder
                        deduped_evidence_set.add(make_sentence_id(evidence[2], evidence[3]))
            evidences[query_id] = deduped_evidence_set

    def generate_samples(query_id, pred_sent_ids):
        curr_pred_evidences = []

        # include all ground truth relevant evidences as positive samples
        for sent_id in evidences[query_id]:
            curr_pred_evidences.append(sent_id)

        # sample negative evidences from pred_sent_ids
        neg_pred_sent_ids = [pred for pred in pred_sent_ids if pred not in evidences[query_id]]
        neg_sent_ids = random.sample(neg_pred_sent_ids, min(len(evidences[query_id]), len(neg_pred_sent_ids)))
        for sent_id in neg_sent_ids:
            doc_id, _ = split_sentence_id(sent_id)
            docs[doc_id] = 'N/A'  # placeholder
            curr_pred_evidences.append(sent_id)

        return curr_pred_evidences

    # read in run file and negative sample using run file ranking predictions
    with open(args.run_file, 'r', encoding='utf-8') as f:
        print('Reading run file...')
        curr_query = None
        pred_sent_ids = []
        for line in f:
            query_id, sent_id, rank = line.strip().split('\t')
            query_id = int(query_id)

            # if we reach a new query in the run file, perform sampling for the previous query
            if query_id != curr_query:
                if curr_query is not None:
                    pred_evidences[curr_query] = generate_samples(curr_query, pred_sent_ids)
                curr_query = query_id
                pred_sent_ids.clear()

            if args.min_rank <= int(rank) <= args.max_rank:
                pred_sent_ids.append(sent_id)

        # perform sampling for the final query
        pred_evidences[curr_query] = generate_samples(curr_query, pred_sent_ids)

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
        for query_id, sent_ids in pred_evidences.items():
            query_text = queries[query_id]

            # only track actual entities that can be found within the query
            actual_entities = []
            for sent_id in evidences[query_id]:
                entity = remove_disambiguation(split_sentence_id(sent_id)[0])
                if entity not in actual_entities and entity.lower() in query_text.lower():
                    actual_entities.append(entity)
            num_actual += len(actual_entities)

            # run NER to get predicted entities
            ner_entities = extract_entities(query_text)
            num_pred += len(ner_entities)

            correct += sum([int(entity.lower() in [ent.lower() for ent in ner_entities]) for entity in actual_entities])

            for rank, sent_id in enumerate(sent_ids):
                relevance = 'true' if sent_id in evidences[query_id] else 'false'
                # get specific sentence from within doc_text
                doc_id, sent_num = split_sentence_id(sent_id)
                entity = doc_id.replace('_', ' ')  # prepend entity name to document text
                doc_text = docs[doc_id]
                sent_text, _ = extract_sentences(doc_text)[sent_num]

                numbered_entities = [f'Entity{i + 1}: {entity}' for i, entity in enumerate(ner_entities)]
                entities_str = ' '.join(numbered_entities)

                f_id.write(f'{query_id}\t{sent_id}\t{rank + 1}\n')
                f_text.write(
                    f'Query: {query_text} Document: {entity} . {normalize_text(sent_text)} {entities_str} Relevant:\t{relevance}\n'
                )

    print('****************************************')
    print(f'Actual Entities: {num_actual}')
    print(f'Predicted Entities: {num_pred}')
    print(f'Correctly Predicted Entities: {correct}')
    print(f'Precision: {correct / num_pred}')
    print(f'Recall: {correct / num_actual}')
    print('****************************************')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates FEVER re-ranking training data with NER entities.')
    parser.add_argument('--dataset_file', required=True, help='FEVER dataset file.')
    parser.add_argument('--run_file', required=True, help='Run file from running retrieval with anserini.')
    parser.add_argument('--collection_folder', required=True, help='FEVER wiki-pages directory.')
    parser.add_argument('--output_id_file', required=True, help='Output query-doc id pairs file.')
    parser.add_argument('--output_text_file', required=True, help='Output query-doc text pairs file.')
    parser.add_argument('--min_rank', type=int, help='Smallest rank to sample from (for negative samples).')
    parser.add_argument('--max_rank', type=int, help='Largest rank to sample from (for negative samples).')
    parser.add_argument('--seed', type=int, help='Optional seed for random sampling.')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    generate_data(args)

    print('Done!')
