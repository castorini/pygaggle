import argparse
import numpy as np

def convert_output(args):
    print('Converting T5 output...')

    with open(args.id_file, 'r', encoding='utf-8') as f_id, open(args.scores_file, 'r', encoding='utf-8') as f_scores, \
            open(args.output_run_file, 'w', encoding='utf-8') as f_run:
        curr_qid = None
        curr_scores = {}
        for id_line, scores_line in zip(f_id, f_scores):
            if args.type == 'mono':
                if args.has_labels:
                    query_id, sent_id, _, _ = id_line.strip().split('\t')
                else:
                    query_id, sent_id, _ = id_line.strip().split('\t')
            else:  # args.type == 'duo'
                if args.has_labels:
                    query_id, sent_id_1, _, sent_id_2, _, _ = id_line.strip().split('\t')
                else:
                    query_id, sent_id_1, _, sent_id_2, _ = id_line.strip().split('\t')
            _, score = scores_line.strip().split('\t')

            # check if we have reached a new query_id
            if query_id != curr_qid:
                # sort previously accumulated doc scores and write to run file
                sorted_scores = sorted(curr_scores.items(), key=lambda x: x[1], reverse=True)
                curr_index = 1
                for curr_sid, curr_score in sorted_scores:
                    if args.k is not None and curr_index > args.k:
                        break
                    # keep the top predicted result even if it does not meet the threshold
                    if curr_index == 1 or args.p is None or np.exp(curr_score) >= args.p:
                        f_run.write(f'{curr_qid}\t{curr_sid}\t{curr_index}\n')
                        curr_index += 1

                # update curr_qid and curr_scores with new query_id
                curr_qid = query_id
                curr_scores.clear()

            # save current score
            if args.type == 'mono':
                curr_scores[sent_id] = float(score)
            else:  # args.type == 'duo'
                if sent_id_1 not in curr_scores:
                    curr_scores[sent_id_1] = 0
                if sent_id_2 not in curr_scores:
                    curr_scores[sent_id_2] = 0
                curr_scores[sent_id_1] += np.exp(float(score))
                curr_scores[sent_id_2] += 1 - np.exp(float(score))

        # write last query_id to file
        sorted_scores = sorted(curr_scores.items(), key=lambda x: x[1], reverse=True)
        curr_index = 1
        for curr_sid, curr_score in sorted_scores:
            if args.k is not None and curr_index > args.k:
                break
            # keep the top predicted result even if it does not meet the threshold
            if curr_index == 1 or args.p is None or np.exp(curr_score) >= args.p:
                f_run.write(f'{curr_qid}\t{curr_sid}\t{curr_index}\n')
                curr_index += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts T5 re-ranking outputs to anserini run file format.')
    parser.add_argument('--id_file', required=True, help='Input query-doc pair ids file.')
    parser.add_argument('--scores_file', required=True, help='Prediction scores file outputted by T5 re-ranking model.')
    parser.add_argument('--output_run_file', required=True, help='Output run file.')
    parser.add_argument('--p', type=float, help='Optional probability threshold.')
    parser.add_argument('--k', type=int, help='Optional top-k cutoff.')
    parser.add_argument('--type', required=True, choices=['mono', 'duo'], help='Type of T5 inference.')
    parser.add_argument('--has_labels', action='store_true', help='Whether the dataset file is labelled.')
    args = parser.parse_args()

    convert_output(args)

    print('Done!')
