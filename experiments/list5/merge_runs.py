import argparse
import itertools

def merge_runs(args):
    rankings = {}

    # read in input run file and save rankings to dict
    for input_file in args.input_run_file:
        with open(input_file, 'r', encoding='utf-8') as f:
            print(f'Reading input run file {input_file}...')
            for line in f:
                query_id, doc_id, _ = line.strip().split('\t')
                if query_id not in rankings:
                    rankings[query_id] = {}
                if input_file not in rankings[query_id]:
                    rankings[query_id][input_file] = []
                rankings[query_id][input_file].append(doc_id)

    # write expanded sentence IDs to output run file
    with open(args.output_run_file, 'w', encoding='utf-8') as f:
        print('Writing merged results to run file...')
        for query_id, files in rankings.items():
            doc_ids = []
            doc_ids_set = set()
            if args.strategy == 'zip':
                for curr_doc_ids in itertools.zip_longest(*files.values()):
                    for doc_id in curr_doc_ids:
                        if doc_id and doc_id not in doc_ids_set:
                            doc_ids.append(doc_id)
                            doc_ids_set.add(doc_id)
            else:  # args.strategy == 'sequential'
                for curr_doc_ids in files.values():
                    for doc_id in curr_doc_ids:
                        if doc_id not in doc_ids_set:
                            doc_ids.append(doc_id)
                            doc_ids_set.add(doc_id)
            for i, doc_id in enumerate(doc_ids):
                f.write(f'{query_id}\t{doc_id}\t{i + 1}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merges several anserini run files.')
    parser.add_argument('--input_run_file', required=True, action='append', help='Input run files.')
    parser.add_argument('--output_run_file', required=True, help='Output run file.')
    parser.add_argument('--strategy', required=True, choices=['zip', 'sequential'], help='Strategy to merge the runs.')
    args = parser.parse_args()

    merge_runs(args)

    print('Done!')
