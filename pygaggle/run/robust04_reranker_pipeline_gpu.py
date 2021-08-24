"""
This script creates monoT5 inputs by taking corpus,
queries and the retrieval run file for the queries.
After that, a monot5-base is fed with these inputs and
outputs a reranked file.

Each line in the output follows TREC Eval format:
    f'{query_id} Q0 {docid} {rank + 1} {1 / (rank + 1)} T5\n'
"""

import argparse
import os
import re
import sys
import spacy
import collections
from tqdm import tqdm
from time import time
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5

def load_corpus(path):
    print('Loading corpus...')
    corpus = {}
    n_text = 0
    with open(path, errors='ignore') as f:
        all_text = ' '.join(f.read().split())
        for raw_text in tqdm(all_text.split('</DOC>')):
            if not raw_text:
                continue
            result = re.search(r'\<DOCNO\>(.*)\<\/DOCNO\>', raw_text)
            if not result:
                continue
            doc_id = result.group(1)
            doc_id = doc_id.strip()

            result = re.search(r'\<TEXT\>(.*)\<\/TEXT\>', raw_text)
            doc_text = ''
            if result:
                doc_text = result.group(1)
                doc_text = doc_text.replace('<P>', ' ').replace('</P>', ' ')
                doc_text = doc_text.strip()
                if doc_text:
                    n_text += 1

            corpus[doc_id] = ' '.join(doc_text.split())

    print(f'Loaded {len(corpus)} docs, {n_text} with texts.')
    return corpus


def load_queries(path):
    """Loads queries into a dict of key: query_id, value: query text."""
    print('Loading queries...')
    queries = {}
    with open(path) as f:
        all_text = ' '.join(f.read().split())

        for query_text in tqdm(all_text.split('</top>')):
            if not query_text:
                continue
            result = re.search(r'\<num\>(.*)\<title\>', query_text)
            query_id = result.group(1)
            query_id = query_id.replace('Number: ', '')
            query_id = ' '.join(query_id.split())

            result = re.search(r'\<title\>(.*)\<desc\>', query_text)
            title = result.group(1)
            title = title.strip()

            result = re.search(r'\<desc\>(.*)\<narr\>', query_text)
            desc = result.group(1)
            desc = desc.replace('Description:', '')
            desc = desc.strip()

            result = re.search(r'\<narr\>(.*)', query_text)
            narr = result.group(1)
            narr = narr.replace('Narrative:', '')
            narr = narr.strip()

            query_text = desc
            if not query_text:
                query_text = title
            queries[query_id] = query_text
    return queries


def load_run(path):
    """Loads run into a dict of key: query_id, value: list of candidate doc
    ids."""
    print('Loading run...')
    run = collections.OrderedDict()
    with open(path) as f:
        for line in tqdm(f):
            query_id, _, doc_title, rank, _, _ = line.split(' ')
            if query_id not in run:
                run[query_id] = []
            run[query_id].append((doc_title, int(rank)))

    # Sort candidate docs by rank.
    sorted_run = collections.OrderedDict()
    for query_id, doc_titles_ranks in run.items():
        sorted(doc_titles_ranks, key=lambda x: x[1])
        doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
        sorted_run[query_id] = doc_titles

    return sorted_run


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Create T5-formatted input from Robust04 queries, run and corpus. After that, it uses a monoT5-base to rerank the candidates. The reranked output is TREC-formatted.')
    parser.add_argument('--queries', type=str, required=True,
                        help='Path to TREC-formatted queries (topics) file.')
    parser.add_argument('--run', type=str, required=True,
                        help='Path to TREC-formatted run file containing the candidates'
                            ' from the first stage retrieval (e.g., BM25).')
    parser.add_argument('--corpus', type=str, required=True,
                        help='path to a single file containing all TREC Disks 4 and 5 documents.')
    parser.add_argument('--output_monot5', type=str, required=True,
                        help='Path to store the TREC-formatted monoT5 reranked output.')
    parser.add_argument('--max_length', type=int, default=10,
                        help='Maximum number of sentences of each segment.')
    parser.add_argument('--stride', type=int, default=5,
                        help='Stride (step) in sentences between each segment.')
    return parser.parse_args(args)

args = parse_args(sys.argv[1:])

# Model
reranker =  MonoT5()
monot5_results = args.output_monot5

# Sentencizer
nlp = spacy.blank("en")
nlp.add_pipe('sentencizer')

# Input Files
queries = load_queries(path=args.queries)
run = load_run(path=args.run)
corpus = load_corpus(path=args.corpus)

# Pipeline
n_segments = 0
n_docs = 0
n_doc_ids_not_found = 0
for query_id, doc_ids in tqdm(run.items(), total=len(run)):
    print(f'{query_id}: Converting to segments...')
    query_text = queries[query_id]
    passages = []
    for doc_id in doc_ids:
        if doc_id not in corpus:
            n_doc_ids_not_found += 1
            continue
        n_docs += 1
        doc_text = corpus[doc_id]
        doc = nlp(doc_text[:10000])
        sentences = [str(sent).strip() for sent in doc.sents]
        for i in range(0, len(sentences), args.stride):
            segment = ' '.join(sentences[i:i + args.max_length])
            passages.append([doc_id, segment])
            n_segments += 1
            if i + args.max_length >= len(sentences):
                break

    print(f'{query_id}: Reranking...')

    # Reranker using pygaggle
    query = Query(query_text)
    texts = [ Text(p[1], {'docid': p[0]}, 0) for p in passages]
    start = time()
    ranked_results = reranker.rerank(query, texts)
    end = time()
    elapsed_time = end - start
    print("Time Elapsed: {:.1f}".format(elapsed_time))

    # Get scores from reranker
    final_t5_scores = {}
    for result in ranked_results:
        if result.metadata["docid"] not in final_t5_scores:
            final_t5_scores[result.metadata["docid"]] = result.score
        else:
            if final_t5_scores[result.metadata["docid"]] < result.score:
                final_t5_scores[result.metadata["docid"]] = result.score

    # Writes a run file in the TREC format
    for rank, (docid, score) in enumerate(final_t5_scores.items()):
        with open(monot5_results, mode='a') as writer:
            writer.write(f'{query_id} Q0 {docid} {rank + 1} {1 / (rank + 1)} T5\n')

print(f'Wrote {n_segments} segments from {n_docs} docs.')
print(f'{n_doc_ids_not_found} doc ids not found in the corpus.')
print('Done!')

