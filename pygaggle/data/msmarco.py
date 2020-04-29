import os
from collections import OrderedDict, defaultdict
from typing import List, Set, OrderedDict, defaultdict
import json
import logging
from itertools import permutations

from pydantic import BaseModel
import scipy.special as sp
import numpy as np

from .relevance import RelevanceExample, Cord19DocumentLoader
from pygaggle.model.tokenize import SpacySenticizer
from pygaggle.rerank.base import Query, Text
from pygaggle.data.unicode import convert_to_unicode


__all__ = ['MsMarcoExample', 'MsMarcoDataset']


class MsMarcoExample(BaseModel):
    qid: str
    text: str
    candidates: List[str]
    relevant_candidates: Set[str]

class MsMarcoDataset(BaseModel):
    examples: List[MsMarcoExample]

    @classmethod
    def load_qrels(cls, path: str) -> DefaultDict[str, Set[str]]:
        qrels = defaultdict(set)
        with open(path) as f:
            for i, line in enumerate(f):
                qid, _, doc_id, _ = line.rstrip().split('\t')
                if int(relevance) >= 1:
                    qrels[qid].add(doc_id)
                if i % 1000 == 0:
                    print(f'Loading qrels {i}')
        return qrels

    @classmethod
    def load_run(cls, path: str) -> OrderedDict[str, List[str]]:
        run = OrderedDict()
        if "eval" in path:
            return run
        with open(path) as f:
            for i, line in enumerate(f):
                qid, doc_title, rank = line.split('\t')
                if qid not in run:
                    run[qid] = []
                run[qid].append((doc_title, int(rank)))
                if i % 1000000 == 0:
                    print(f'Loading run {i}')
        sorted_run = OrderedDict()
        for qid, doc_titles_ranks in run.items():
            sorted(doc_titles_ranks, key=lambda x: x[1])
            doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
            sorted_run[qid] = doc_titles
        return sorted_run

    @classmethod
    def load_queries(cls, 
                     path: str, 
                     qrels: DefaultDict[str, Set[str]], 
                     run: OrderedDict[str, List[str]]) -> List[MsMarcoExample]:
        queries = []
        with open(path) as f:
            for i, line in enumerate(f):
                qid, query = line.rstrip().split('\t')
                try:
                    candidates = run[qid]
                queries.append(MsMarcoExample(qid = qid,
                                              query = query,
                                              candidates = run[qid],
                                              relevant_candidates = qrels[qid]))
                if i % 1000 == 0:
                    print(f'Loading queries {i}')
        return queries

    @classmethod
    def from_folder(cls, 
                    folder: str, 
                    split: str = 'dev', 
                    is_duo: bool = False) -> 'MsMarcoDataset':
        with open(filename) as f:
            run_mono = "mono." if is_duo else ""
            query_path = os.path.join(f"queries.{split}.small.tsv")
            qrels_path = os.path.join(f"qrels.{split}.small.tsv")
            run_path = os.path.join(f"run.{run_mono}{split}.small.tsv")
            return cls(cls.load_queries(query_path, 
                                        cls.load_qrels(qrels_path),
                                        cls.load_run(run_path)))


    def query_passage_tuples(self, is_duo: bool = False):
        return (((ex.qid, ex.text, ex.relevant_candidates), perm_pas) for ex in self.examples
                for perm_pas in permutations(ex.candidates, r=1+int(is_duo)))

    
    def to_relevance_examples(self,
                              index_path: str,
                              is_duo: bool = False) -> List[RelevanceExample]:
        loader = MsMarcoPassageLoader(index_path)
        example_map = {}
        for (qid, text, rel_cands), cands in self.query_answer_pairs():
            if qid not in example_map:
                example_map[qid] = [convert_to_unicode(text), [], [], []]
            #TODO generalize for duoBERT by removing indexing to 0
            example_map[qid][1].append([cand for cand in cands][0])
            try:
                passages = [loader.load_passage(cand) for cand in cands]
                #TODO generalize for duoBERT by removing indexing to 0
                example_map[qid][2].append([convert_to_unicode(passage.all_text) for passage in passages][0]) 
            except ValueError as e:
                logging.warning(f'Skipping {passages}')
                continue
            #TODO generalize for duoBERT by adding indexing cands to 0
            example_map[qid][3].append(cands in rel_cands)
        mean_stats = defaultdict(list)
        for ex in self.examples:
            int_rels = np.array(list(map(int, example_map[ex.qid][3])))
            p = int_rels.sum()/(len(candidates) - 1) if is_duo else int_rels.sum()
            mean_stats['Random P@1'].append(np.mean(int_rels))
            n = len(candidates) - p
            N = len(candidates)
            if len(candidates) <= 1000:
                mean_stats['Random R@1000'].append(1 if 1 in int_rels else 0)
            numer = np.array([sp.comb(n, i) / (N - i) for i in range(0, n + 1)]) * p
            denom = np.array([sp.comb(N, i) for i in range(0, n + 1)])
            rr = 1 / np.arange(1, n + 2)
            rmrr = np.sum(numer * rr / denom)
            mean_stats['Random MRR'].append(rmrr)
            rmrr10 = np.sum(numer[:10] * rr[:10] / denom[:10]) #TODO verify if this works
            mean_stats['Random MRR@10'].append(rmmr10)
            ex_index = len(candidates)
            for rel_cand in ex.relevant_candidates:
                if rel_cand in ex.candidates:
                    ex_index = min(ex.candidates.index(rel_cand), ex_index)
            mean_stats['Existing MRR'].append(1 / (ex_index + 1) if ex_index < len(candidates) else 0)
            mean_stats['Existing MRR@10'].append(1 / (ex_index + 1) if ex_index < 10 else 0)
        for k, v in mean_stats.items():
            logging.info(f'{k}: {np.mean(v)}')
        return [RelevanceExample(Query(query_text, qid), 
                                 list(map(lambda s: Text(s[0], dict(docid=key[1])), zip(cands, cands_text))), 
                                 rel_cands) \
                for qid, (query_text, cands, cands_text, rel_cands) in example_map.items()]
