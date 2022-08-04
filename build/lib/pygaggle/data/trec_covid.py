import os
from collections import OrderedDict, defaultdict
from typing import List, Set, DefaultDict
import logging
from itertools import permutations

from pydantic import BaseModel
import scipy.special as sp
import numpy as np
from tqdm import tqdm

from .relevance import RelevanceExample, Cord19AbstractLoader
from pygaggle.rerank.base import Query, Text
from pygaggle.data.unicode import convert_to_unicode
import xml.etree.ElementTree as ElementTree


__all__ = ['TRECCovidExample', 'TRECCovidDataset']


class TRECCovidExample(BaseModel):
    qid: str
    text: str
    candidates: List[str]
    relevant_candidates: Set[str]


class TRECCovidDataset(BaseModel):
    examples: List[TRECCovidExample]

    @classmethod
    def load_qrels(cls, path: str) -> DefaultDict[str, Set[str]]:
        qrels = defaultdict(set)
        with open(path) as f:
            for _, line in enumerate(f):
                qid, _, doc_id, relevance = filter(None, line.rstrip().split(' '))
                if int(relevance) >= 1:
                    qrels[qid].add(doc_id)
        return qrels

    @classmethod
    def load_run(cls, path: str):
        '''Returns OrderedDict[str, List[str]]'''
        run = OrderedDict()
        with open(path) as f:
            for _, line in enumerate(f):
                # Line is of the format {qid}, QO, {docid}, {rank}, {score},
                # {tag}.
                qid, _, doc_title, rank, score, _ = line.split(' ')
                if qid not in run:
                    run[qid] = []
                run[qid].append((doc_title, int(rank)))
        sorted_run = OrderedDict()
        for qid, doc_titles_ranks in run.items():
            doc_titles_ranks.sort(key=lambda x: x[1])
            doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
            sorted_run[qid] = doc_titles
        return sorted_run

    # Should this be queries or topics?
    @classmethod
    def load_queries(cls,
                     path: str,
                     qrels: DefaultDict[str, Set[str]],
                     run) -> List[TRECCovidExample]:
        queries = []
        query_xml_tree = ElementTree.parse(path)
        for topic in query_xml_tree.getroot():
            qid = topic.attrib["number"]
            query = topic.find("query").text
            queries.append(
                TRECCovidExample(
                    qid=qid,
                    text=query,
                    candidates=run[qid],
                    relevant_candidates=qrels[qid]))
        return queries

    @classmethod
    def from_folder(cls,
                    folder: str) -> 'TRECCovidDataset':
        query_path = os.path.join(folder, "topics.covid-round5.xml")
        qrels_path = os.path.join(folder, "qrels-covid_d5_j4.5-5.txt")
        run_path = os.path.join(folder, "expanded.anserini.final-r5.fusion1.txt")
        return cls(examples=cls.load_queries(query_path,
                                             cls.load_qrels(qrels_path),
                                             cls.load_run(run_path)))

    def query_document_tuples(self):
        return [((ex.qid, ex.text, ex.relevant_candidates), perm_pas)
                for ex in self.examples
                for perm_pas in permutations(ex.candidates, r=1)]

    def to_relevance_examples(self,
                              index_path: str) -> List[RelevanceExample]:
        loader = Cord19AbstractLoader(index_path)
        example_map = {}
        for (qid, text, rel_cands), cands in tqdm(self.query_document_tuples()):
            if qid not in example_map:
                example_map[qid] = [convert_to_unicode(text), [], [], [], []]
            example_map[qid][1].append([cand for cand in cands][0])
            try:
                passages = [loader.load_document(cand) for cand in cands]
                # Sometimes this abstract is empty.
                example_map[qid][2].append(
                    [convert_to_unicode(passage.abstract)
                     for passage in passages][0])
                example_map[qid][4].append(
                    [convert_to_unicode(passage.title)
                     for passage in passages][0])
            except ValueError as e:
                logging.error(e)
                logging.warning('Skipping passages')
                continue
            example_map[qid][3].append(cands[0] in rel_cands)
        mean_stats = defaultdict(list)

        for ex in self.examples:
            int_rels = np.array(list(map(int, example_map[ex.qid][3])))
            p = int(int_rels.sum())
            mean_stats['Expected P@1 for Random Ordering'].append(np.mean(int_rels))
            n = len(ex.candidates) - p
            N = len(ex.candidates)
            if len(ex.candidates) <= 1000:
                mean_stats['Expected R@1000 for Random Ordering'].append(1 if 1 in int_rels else 0)
            numer = np.array([sp.comb(n, i) / (N - i) for i in range(0, n + 1) if i != N]) * p
            if n == N:
                numer = np.append(numer, 0)
            denom = np.array([sp.comb(N, i) for i in range(0, n + 1)])
            rr = 1 / np.arange(1, n + 2)
            rmrr = np.sum(numer * rr / denom)
            mean_stats['Expected MRR for Random Ordering'].append(rmrr)
            rmrr10 = np.sum(numer[:10] * rr[:10] / denom[:10])
            mean_stats['Expected MRR@10 for Random Ordering'].append(rmrr10)
            ex_index = len(ex.candidates)
            for rel_cand in ex.relevant_candidates:
                if rel_cand in ex.candidates:
                    ex_index = min(ex.candidates.index(rel_cand), ex_index)
            mean_stats['Existing MRR'].append(1 / (ex_index + 1)
                                              if ex_index < len(ex.candidates)
                                              else 0)
            mean_stats['Existing MRR@10'].append(1 / (ex_index + 1) if ex_index < 10 else 0)
        for k, v in mean_stats.items():
            logging.info(f'{k}: {np.mean(v)}')
        rel = [RelevanceExample(Query(text=query_text, id=qid),
                                list(map(lambda s: Text(s[1], dict(docid=s[0]), title=s[2]),
                                         zip(cands, cands_text, title))), rel_cands)
               for qid, (query_text, cands, cands_text, rel_cands, title) in example_map.items()]
        return rel
