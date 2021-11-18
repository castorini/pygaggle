"""
This script creates monoT5 input files by taking corpus,
queries and the retrieval run file for the queries and then
create files for monoT5 input. Each line in the duoT5 input
file follows the format:
    f'Query: {query} Document0: {document0} Document1: {document1} Relevant:\n')
"""
import collections
from tqdm import tqdm
import argparse
from pyserini.search import SimpleSearcher
import spacy
import ujson as json
import re


parser = argparse.ArgumentParser()
parser.add_argument("--queries", type=str, required=True,
                    help="tsv file with two columns, <query_id> and <query_text>")
parser.add_argument("--run", type=str, required=True,
                    help="tsv file with three columns <query_id>, <doc_id> and <rank>")
parser.add_argument("--index", type=str, required=True)
parser.add_argument("--stride", type=int, required=True)
parser.add_argument("--length", type=int, required=True)
parser.add_argument("--t5_input", type=str, required=True,
                    help="path to store t5_input, txt format")
parser.add_argument("--t5_input_ids", type=str, required=True,
                    help="path to store the query-doc ids of t5_input, tsv format")
parser.add_argument("--year", type=int, default=2020,
                    help="path to store the query-doc ids of t5_input, tsv format")
args = parser.parse_args()


def load_queries(path):
    """Loads queries into a dict of key: query_id, value: query text."""
    print('Loading queries...')
    queries = {}
    with open(path) as f:
        for line in tqdm(f):
            query_id, query = line.rstrip().split("\t")
            queries[query_id] = query
    return queries


def load_run(path, topk=1000):
    """Loads run into a dict of key: query_id, value: list of candidate doc
    ids."""

    # We want to preserve the order of runs so we can pair the run file with
    # the TFRecord file.
    print('Loading run...')
    run = collections.OrderedDict()
    with open(path) as f:
        for line in tqdm(f):
            query_id, _, doc_title, rank, _, _ = line.split()
            if query_id not in run:
                run[query_id] = []
            run[query_id].append((doc_title, int(rank)))

    # Sort candidate docs by rank.
    print('Sorting candidate docs by rank...')
    sorted_run = collections.OrderedDict()
    for query_id, doc_titles_ranks in tqdm(run.items()):
        doc_titles_ranks.sort(key=lambda x: x[1])
        doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks[:topk]]
        sorted_run[query_id] = doc_titles

    return sorted_run

index = SimpleSearcher(args.index)
queries = load_queries(path=args.queries)
run = load_run(path=args.run)
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

print("Writing t5 input and ids")
with open(args.t5_input, 'w') as fout_t5, open(args.t5_input_ids, 'w') as fout_tsv:
    for num_examples, (query_id, candidate_doc_ids) in enumerate(
            tqdm(run.items(), total=len(run))):
        query = queries[query_id]
        seen = {}
        for candidate_doc_id in candidate_doc_ids:
            if candidate_doc_id.split("#")[0] in seen:
                passage, ind_desc = seen[candidate_doc_id.split("#")[0]]
                candidate_doc_id = candidate_doc_id.split("#")[0]
            else:
                if args.year == 2020:
                    try:
                        candidate_doc_id, ind_desc = candidate_doc_id.split("#")
                        content = index.doc(f"<urn:uuid:{candidate_doc_id}>").contents()
                        ind_desc = int(ind_desc)
                    except:
                        print(candidate_doc_id)
                        content = ""
                        ind_desc = 0
                elif args.year == 2021:
                    candidate_doc_id, ind_desc = candidate_doc_id.split("#")
                    ind_desc = int(ind_desc)
                    content = json.loads(index.doc(str(candidate_doc_id)).raw())
                    content = content['text']
                else:
                    try:
                        candidate_doc_id, ind_desc = candidate_doc_id.split("#")
                        ind_desc = int(ind_desc)
                        content = index.doc(candidate_doc_id).raw()
                    except:
                        print(candidate_doc_id)
                        content = ""
                        ind_desc = 0
                content = re.sub('\s+'," ", content)
                all_sentences_description = [sent.sent.text.strip() for sent in nlp(f"{content}").sents]

                example = " ".join(all_sentences_description[ind_desc: ind_desc + args.length])
                passage = f"{example}"
                passage = re.sub('\s+', ' ', passage.strip())
                seen[candidate_doc_id] = (passage, ind_desc)
                fout_t5.write(
                    f'Query: {query} Document: {passage} Relevant:\n')
                fout_tsv.write(f'{query_id}\t{candidate_doc_id}#{ind_desc}\n')
