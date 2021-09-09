import argparse
import json

def calculate_scores(args):
    evidences = {}
    labels = {}

    total = 0
    correct = 0
    strict_correct = 0
    total_hits = 0
    total_precision = 0
    total_recall = 0

    with open(args.dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_json = json.loads(line.strip())
            query_id = line_json['id']
            if 'label' in line_json:  # no "label" field in test datasets
                label = line_json['label']
                labels[query_id] = label

                if label not in ['SUPPORTS', 'REFUTES']:
                    continue

            evidence_sets = []
            for annotator in line_json['evidence']:
                evidence_set = [[evidence[2], evidence[3]] for evidence in annotator]
                evidence_sets.append(evidence_set)
            evidences[query_id] = evidence_sets

    def check_evidence_set(pred_evidence_set, true_evidence_sets):
        for evidence_set in true_evidence_sets:
            if all([evidence in pred_evidence_set for evidence in evidence_set]):
                return True

        return False

    with open(args.submission_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_json = json.loads(line.strip())
            query_id = line_json['id']
            pred_label = line_json['predicted_label']
            pred_evidence_set = line_json['predicted_evidence']

            total += 1
            if pred_label == labels[query_id]:
                correct += 1
                if labels[query_id] == 'NOT ENOUGH INFO' or check_evidence_set(pred_evidence_set, evidences[query_id]):
                    strict_correct += 1

            if labels[query_id] != 'NOT ENOUGH INFO':
                total_hits += 1

                # calculate precision
                correct_evidence = [ev for ev_set in evidences[query_id] for ev in ev_set if ev[1] is not None]
                if len(pred_evidence_set) == 0:
                    total_precision += 1
                else:
                    curr_precision = 0
                    curr_precision_hits = 0
                    for pred in pred_evidence_set:
                        curr_precision_hits += 1
                        if pred in correct_evidence:
                            curr_precision += 1
                    total_precision += curr_precision / curr_precision_hits

                # calculate recall
                if len(evidences[query_id]) == 0 or all([len(ev_set) == 0 for ev_set in evidences[query_id]]) or \
                        check_evidence_set(pred_evidence_set, evidences[query_id]):
                    total_recall += 1

    print('****************************************')
    fever_score = strict_correct / total
    print(f'FEVER Score: {fever_score}')
    label_acc = correct / total
    print(f'Label Accuracy: {label_acc}')
    precision = (total_precision / total_hits) if total_hits > 0 else 1.0
    print(f'Evidence Precision: {precision}')
    recall = (total_recall / total_hits) if total_hits > 0 else 0.0
    print(f'Evidence Recall: {recall}')
    f1 = 2.0 * precision * recall / (precision + recall)
    print(f'Evidence F1: {f1}')
    print('****************************************')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates various metrics used in the FEVER shared task.')
    parser.add_argument('--dataset_file', required=True, help='FEVER dataset file.')
    parser.add_argument('--submission_file', required=True, help='Submission file to FEVER shared task.')
    args = parser.parse_args()

    calculate_scores(args)
