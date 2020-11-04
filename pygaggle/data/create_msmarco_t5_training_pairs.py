"""
This script creates monoT5 input files for training,
Each line in the monoT5 input file follows the format:
    f'Query: {query} Document: {document} Relevant:\t{label}\n')
"""
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--triples_train", type=str, required=True,
                    help="tsv file <query>, <positive_document>, <negative_document>")
parser.add_argument("--output_to_t5", type=str, required=True,
                    help="t5 train input file")
args = parser.parse_args()

with open(args.output_to_t5, 'w') as fout_t5:
    for line_num, line in enumerate(tqdm(open(args.triples_train))):
        query, positive_document, negative_document = line.strip().split('\t')
        fout_t5.write(f'Query: {query} Document: {positive_document} Relevant:\ttrue\n')
        fout_t5.write(f'Query: {query} Document: {negative_document} Relevant:\tfalse\n')
print('Done!')
