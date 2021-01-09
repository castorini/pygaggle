"""
This script creates duoT5 input files by taking corpus,
queries and the pointwise ranking run file for the queries and then
create files for duoT5 input. Each line in the duoT5 input
file follows the format:
    f'Query: {query} Document0: {document0} Document1: {document1} Relevant:\n')
"""
import collections
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--queries", type=str, required=True,
                    help="tsv file with two columns, <query_id> and <query_text>")
parser.add_argument("--run", type=str, required=True,
                    help="tsv file with five columns <query_id>, <doc_id_a>, <doc_id_b>, <rank_a> and <rank_b>")
parser.add_argument("--corpus", type=str, required=True)
parser.add_argument("--t5_input", type=str, required=True,
                    help="path to store t5_input, txt format")
parser.add_argument("--t5_input_ids", type=str, required=True,
                    help="path to store the query-doc ids of t5_input, tsv format")
parser.add_argument("--top_k", type=int, default=50,
                    help="top-k pointwise hits to be reranked by pairwise ranker")
args = parser.parse_args()


def load_corpus(path):
    print('Loading corpus...')
    corpus = {}
    with open(path) as f:
        for line in tqdm(f):
            doc_id, doc = line.rstrip().split('\t')
            corpus[doc_id] = doc
    return corpus


def load_queries(path):
    """Loads queries into a dict of key: query_id, value: query text."""
    print('Loading queries...')
    queries = {}
    with open(path) as f:
        for line in tqdm(f):
            query_id, query = line.rstrip().split('\t')
            queries[query_id] = query
    return queries


def load_run(path, top_k=50):
    """Loads run into a dict of key: query_id, value: list of candidate doc
    ids."""

    # We want to preserve the order of runs so we can pair the run file with
    # the TFRecord file.
    print('Loading run...')
    run = collections.OrderedDict()
    with open(path) as f:
        for line in tqdm(f):
            query_id, doc_title, rank = line.split('\t')
            if query_id not in run:
                run[query_id] = []
            run[query_id].append((doc_title, int(rank)))

    # Sort candidate docs by rank.
    print('Sorting candidate docs by rank...')
    sorted_run = collections.OrderedDict()
    for query_id, doc_titles_ranks in tqdm(run.items()):
        sorted(doc_titles_ranks, key=lambda x: x[1])
        doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks][:top_k]
        sorted_run[query_id] = doc_titles

    return sorted_run


corpus = load_corpus(path=args.corpus)
queries = load_queries(path=args.queries)
run = load_run(path=args.run, top_k=args.top_k)

print("Writing t5 input and ids")
with open(args.t5_input, 'w') as fout_t5, open(args.t5_input_ids, 'w') as fout_tsv:
    for num_examples, (query_id, candidate_doc_ids) in enumerate(
            tqdm(run.items(), total=len(run))):
        query = queries[query_id]
        for ct_a, candidate_doc_id_a in enumerate(candidate_doc_ids):
            for ct_b, candidate_doc_id_b in enumerate(candidate_doc_ids):
                if candidate_doc_id_a == candidate_doc_id_b:
                    continue
                fout_t5.write(
                    f'Query: {query} Document0: {corpus[candidate_doc_id_a]} '
                    f'Document1: {corpus[candidate_doc_id_b]} Relevant:\n'
                )
                fout_tsv.write(f'{query_id}\t{candidate_doc_id_a}\t{candidate_doc_id_b}\t{ct_a}\t{ct_b}\n')
