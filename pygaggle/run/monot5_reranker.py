import argparse
import pandas as pd
from tqdm import tqdm
from pygaggle.rerank.transformer import MonoT5
from pygaggle.rerank.base import Query, Text
import jsonlines


def load_corpus(path):
    print('Loading corpus...')
    corpus = {}
    if '.json' in path:
        with jsonlines.open(path) as reader:
            for obj in tqdm(reader):
                id = int(obj['id'])
                corpus[id] = obj['contents']
    else: #Assume it's a .tsv
        corpus = pd.read_csv(path, sep='\t', header=None, index_col=0)[1].to_dict()
    return corpus


def load_run(path):
    print('Loading run...')
    run = pd.read_csv(path, delim_whitespace=True, header=None)
    run = run.groupby(0)[1].apply(list).to_dict()
    return run


def load_queries(path):
    print('Loading queries...')
    queries = pd.read_csv(path, sep='\t', header=None, index_col=0)
    queries = queries[1].to_dict()
    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default='unicamp-dl/mt5-base-multi-msmarco', type=str, required=False,
                        help="Reranker model.")
    parser.add_argument("--initial_run", default=None, type=str, required=True,
                        help="Initial run to be reranked.")
    parser.add_argument("--corpus", default=None, type=str, required=True,
                        help="Document collection.")
    parser.add_argument("--output_run", default=None, type=str, required=True,
                        help="Path to save the reranked run.")
    parser.add_argument("--queries", default=None, type=str, required=True,
                        help="Path to the queries file.")

    args = parser.parse_args()
    model = MonoT5(args.model_name_or_path)
    run = load_run(args.initial_run)
    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)

    # Run reranker
    trec = open(args.output_run + '-trec.txt','w')
    marco = open(args.output_run + '-marco.txt','w')
    for idx, query_id in enumerate(tqdm(run.keys())):
        query = Query(queries[query_id])
        texts = [Text(corpus[doc_id], {'docid': doc_id}, 0) for doc_id in run[query_id]]
        reranked = model.rerank(query, texts)
        for rank, document in enumerate(reranked):
            trec.write(f'{query_id}\tQ0\t{document.metadata["docid"]}\t{rank+1}\t{document.score}\t{args.model_name_or_path}\n')
            marco.write(f'{query_id}\t{document.metadata["docid"]}\t{rank+1}\n')
    trec.close()
    marco.close()
    print("Done!")
if __name__ == "__main__":
    main()