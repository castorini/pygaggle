import argparse
import glob
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from fever_utils import make_sentence_id

def calculate_scores(args):
    evidences = {}

    with open(args.dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_json = json.loads(line.strip())

            evidence_sets = []
            if line_json['label'] != 'NOT ENOUGH INFO':
                for annotator in line_json['evidence']:
                    evidence_set = [make_sentence_id(evidence[2], evidence[3]) for evidence in annotator]
                    evidence_sets.append(evidence_set)
            evidences[line_json['id']] = evidence_sets

    def aggregate(scores):
        if args.num_classes == 4:
            # filter out samples predicted weak and remove weak scores
            scores = scores[np.argmax(scores, axis=1) != 3][:, :3]
            if len(scores) == 0:
                return 1

        if args.strategy == 'first':
            return np.argmax(scores[0])
        elif args.strategy == 'sum':
            return np.argmax(np.sum(np.exp(scores), axis=0))
        elif args.strategy == 'nei_default':
            maxes = np.argmax(scores, axis=1)
            if (0 in maxes and 2 in maxes) or (0 not in maxes and 2 not in maxes):
                return 1
            elif 0 in maxes:
                return 0
            elif 2 in maxes:
                return 2
            return -1
        elif args.strategy == 'max':
            return np.argmax(np.max(np.exp(scores), axis=0))
        return -1

    for scores_file in sorted(glob.glob(f'{args.scores_files_prefix}*')):
        labels = []
        pred_labels = []
        fever_scores = []
        with open(args.id_file, 'r', encoding='utf-8') as f_id, open(scores_file, 'r', encoding='utf-8') as f_scores:
            curr_query = None
            curr_label = None  # actual label for current query
            curr_scores = []
            curr_evidences = []
            for id_line, scores_line in zip(f_id, f_scores):
                query_id, sent_ids, label_str = id_line.strip().split('\t')
                query_id = int(query_id)

                if query_id != curr_query:
                    if curr_query is not None:
                        # aggregate to get predicted label
                        pred_label = aggregate(np.array(curr_scores))
                        pred_labels.append(pred_label)
                        # calculate FEVER score
                        fever_scores.append(int(pred_label == curr_label and (pred_label == 1 or \
                                any([set(ev_set).issubset(set(curr_evidences)) for ev_set in evidences[curr_query]]))))
                    curr_query = query_id
                    curr_scores.clear()
                    curr_evidences.clear()
                    # save actual label
                    if label_str == 'false':
                        curr_label = 0
                    elif label_str == 'weak':
                        curr_label = 1
                    elif label_str == 'true':
                        curr_label = 2
                    labels.append(curr_label)

                # save predicted evidence(s) and scores
                if args.num_classes == 3:
                    _, false_score, nei_score, true_score = scores_line.strip().split('\t')
                    scores = [float(false_score), float(nei_score), float(true_score)]
                elif args.num_classes == 4:
                    _, false_score, ignore_score, true_score, nei_score = scores_line.strip().split('\t')
                    scores = [float(false_score), float(nei_score), float(true_score), float(ignore_score)]
                curr_scores.append(scores)
                curr_evidences.extend(sent_ids.strip().split(' '))

            # handle last query
            pred_label = aggregate(np.array(curr_scores))
            pred_labels.append(pred_label)
            fever_scores.append(int(pred_label == curr_label and (pred_label == 1 or \
                    any([set(ev_set).issubset(set(curr_evidences)) for ev_set in evidences[curr_query]]))))

            print(scores_file)
            print(f'Label Accuracy: {accuracy_score(labels, pred_labels)}')
            print(f'Predicted Label F1 Scores: {f1_score(labels, pred_labels, average=None)}')
            print(f'Predicted Label Distribution: {[pred_labels.count(i) for i in range(args.num_classes)]}')
            print(f'FEVER Score: {sum(fever_scores) / len(fever_scores)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates various metrics of label prediction output files.')
    parser.add_argument('--id_file', required=True, help='Input query-doc pair ids file.')
    parser.add_argument('--scores_files_prefix', required=True, help='Prefix of all T5 label prediction scores files.')
    parser.add_argument('--dataset_file', help='FEVER dataset file.')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of label prediction classes.')
    parser.add_argument('--strategy', help='Format of scores file and method of aggregation if applicable.')
    args = parser.parse_args()

    calculate_scores(args)
