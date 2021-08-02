import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_file', required=True, help='UKP-Athene doc retrieval output file.')
parser.add_argument('--output_run_file', required=True, help='Output run file.')
args = parser.parse_args()

with open(args.dataset_file, 'r', encoding='utf-8') as f_in, open(args.output_run_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        i = 0
        line_json = json.loads(line.strip())
        qid = line_json['id']

        for did in line_json['predicted_pages']:
            i += 1
            f_out.write(f'{qid}\t{did}\t{i}\n')
