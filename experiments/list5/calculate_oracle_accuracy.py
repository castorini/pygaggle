import argparse
from collections import Counter
import ftfy
import json

from fever_utils import make_sentence_id

def calculate_stats(args):
    evidences = {}
    num_evidences = []

    correct = {cutoff: 0 for cutoff in args.k}
    max_cutoff = max(args.k)
    num_verifiable_queries = 0
    num_queries = 0

    # read in dataset file and save evidences to dicts
    with open(args.dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_json = json.loads(line.strip())

            query_id = line_json['id']
            label = line_json['label']
            if label != 'NOT ENOUGH INFO':
                num_verifiable_queries += 1
            num_queries += 1

            annotators = []
            # no evidence set for NEI queries
            if label != 'NOT ENOUGH INFO':
                for annotator in line_json['evidence']:
                    evidence_set = []
                    for evidence in annotator:
                        evidence[2] = ftfy.fix_text(evidence[2])
                        evidence_set.append(make_sentence_id(evidence[2], evidence[3]))
                    annotators.append(evidence_set)
                num_evidences.append(min([len(evidence_set) for evidence_set in annotators]))

            evidences[query_id] = annotators

    # read in run file and record cutoff counts
    with open(args.run_file, 'r', encoding='utf-8') as f:
        curr_query = None
        pred_sent_ids = []
        for line in f:
            query_id, sent_id, rank = line.strip().split('\t')
            query_id = int(query_id)

            if query_id != curr_query:
                if curr_query is not None:
                    for rank in args.k:
                        if not evidences[curr_query]:  # if query is NEI, assume it is correct
                            correct[rank] += 1
                        else:
                            for evidence_set in evidences[curr_query]:
                                if all([evidence in pred_sent_ids for evidence in evidence_set]):
                                    correct[rank] += 1
                                    break
                curr_query = query_id
                pred_sent_ids.clear()

            if int(rank) <= max_cutoff:
                pred_sent_ids.append(sent_id)

        # handle last query
        for rank in args.k:
            if not evidences[curr_query]:  # if query is NEI, assume it is correct
                correct[rank] += 1
            else:
                for evidence_set in evidences[curr_query]:
                    if all([evidence in pred_sent_ids for evidence in evidence_set]):
                        correct[rank] += 1
                        break

    # print number of queries that can be verified with each minimum number of evidences
    evidences_counter = Counter(num_evidences)
    for num_evidence, count in evidences_counter.most_common():
        print(f'{num_evidence}-verifiable queries: {count / num_verifiable_queries}')
    print('--------------------------------------------------')
    # print oracle accuracies
    for cutoff, num_correct in correct.items():
        print(f'Oracle accuracy for top {cutoff}: {num_correct / num_queries}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculates oracle accuracy (upper bound for label prediction).')
    parser.add_argument('--dataset_file', required=True, help='FEVER dataset file.')
    parser.add_argument('--run_file', required=True, help='Run file generated after re-ranking.')
    parser.add_argument('--k', nargs='+', type=int, help='Cutoff values to calculate oracle accuracy for.')
    args = parser.parse_args()

    calculate_stats(args)