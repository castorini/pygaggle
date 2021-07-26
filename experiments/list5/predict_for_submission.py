import argparse
import csv
import json
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from fever_utils import split_sentence_id

def predict(args):
    preds = {}

    def aggregate(query_id, scores, sent_ids):
        pred = {}

        best = np.argmax(scores[0])

        pred['id'] = query_id
        if best == 0:
            pred['predicted_label'] = 'REFUTES'
        elif best == 1:
            pred['predicted_label'] = 'NOT ENOUGH INFO'
        else:  # best == 2
            pred['predicted_label'] = 'SUPPORTS'
        pred['predicted_evidence'] = [list(split_sentence_id(sent)) for sent in sent_ids]

        return best, pred

    with open(args.id_file, 'r', encoding='utf-8') as f_id, open(args.scores_file, 'r', encoding='utf-8') as f_scores, \
            open(args.output_predictions_file, 'w', encoding='utf-8') as f_out:
        print('Reading scores file...')
        curr_query = None
        curr_sent_ids = []
        curr_scores = []
        for id_line, scores_line in zip(f_id, f_scores):
            if args.has_labels:
                query_id, sent_ids, label = id_line.strip().split('\t')
            else:
                query_id, sent_ids = id_line.strip().split('\t')
            query_id = int(query_id)
            _, false_score, nei_score, true_score = scores_line.strip().split('\t')

            if query_id != curr_query:
                if curr_query is not None:
                    best, pred = aggregate(curr_query, curr_scores, curr_sent_ids)
                    json.dump(pred, f_out)
                    f_out.write('\n')
                    preds[curr_query] = best
                curr_query = query_id
                curr_sent_ids.clear()
                curr_scores.clear()

            curr_sent_ids = sent_ids.split(' ')
            curr_scores.append((float(false_score), float(nei_score), float(true_score)))

        best, pred = aggregate(curr_query, curr_scores, curr_sent_ids)
        json.dump(pred, f_out)
        f_out.write('\n')
        preds[curr_query] = best

    # print label prediction metrics if dataset file provided
    if args.dataset_file:
        actual_labels = []
        pred_labels = []
        with open(args.dataset_file, 'r', encoding='utf-8') as f:
            print('Reading FEVER dataset file...')
            for line in f:
                line_json = json.loads(line.strip())

                label = line_json['label']
                if label == 'SUPPORTS':
                    actual_labels.append(2)
                elif label == 'REFUTES':
                    actual_labels.append(0)
                else:  # label == 'NOT ENOUGH INFO'
                    actual_labels.append(1)

                query_id = line_json['id']
                pred_labels.append(preds[query_id])

        print('****************************************')
        print(f'Number of Queries: {len(actual_labels)}')
        print(f'Label Accuracy: {accuracy_score(actual_labels, pred_labels)}')
        print(f'Label F1: {f1_score(actual_labels, pred_labels, average=None)}')
        print(confusion_matrix(actual_labels, pred_labels))
        print('****************************************')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicts labels and evidence sentences for FEVER submission.')
    parser.add_argument('--id_file', required=True, help='Input query-doc pair ids file.')
    parser.add_argument('--scores_file',
                        required=True,
                        help='Prediction scores file outputted by T5 label prediction model.')
    parser.add_argument('--dataset_file', help='FEVER dataset file (only if labelled).')
    parser.add_argument('--output_predictions_file',
                        required=True,
                        help='Output predictions file in FEVER submission format.')
    parser.add_argument('--evidence_k',
                        type=int,
                        default=5,
                        help='Number of top sentences to use as evidence for FEVER submission.')
    parser.add_argument('--has_labels', action='store_true', help='Whether the id file is labelled.')
    args = parser.parse_args()

    predict(args)

    print('Done!')
