from collections import Counter
from copy import deepcopy
from typing import List
import math

from pyserini.analysis import get_lucene_analyzer, Analyzer
from pyserini.index import IndexReader
import numpy as np

from .base import Reranker, Query, Text


__all__ = ['Bm25Reranker']


class Bm25Reranker(Reranker):
    def __init__(self,
                 k1: float = 1.6,
                 b: float = 0.75,
                 index_path: str = None):
        self.k1 = k1
        self.b = b
        self.use_corpus_estimator = False
        self.analyzer = Analyzer(get_lucene_analyzer())
        if index_path:
            self.use_corpus_estimator = True
            self.index_utils = IndexReader(index_path)

    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        query_words = self.analyzer.analyze(query.text)
        sentences = list(map(self.analyzer.analyze, (t.text for t in texts)))

        query_words_set = set(query_words)
        sentence_sets = list(map(set, sentences))
        if not self.use_corpus_estimator:
            idfs = {w: math.log(len(sentence_sets) / (1 + sum(int(w in sent)
                    for sent in sentence_sets)))
                    for w in query_words_set}
        mean_len = np.mean(list(map(len, sentences)))
        d_len = len(sentences)

        texts = deepcopy(texts)
        for sent_words, text in zip(sentences, texts):
            tf = Counter(filter(query_words.__contains__, sent_words))
            if self.use_corpus_estimator:
                idfs = {w:
                        self.index_utils.compute_bm25_term_weight(
                            text.metadata['docid'], w) for w in tf}
            score = sum(idfs[w] * tf[w] * (self.k1 + 1) /
                        (tf[w] + self.k1 * (1 - self.b + self.b *
                                            (d_len / mean_len))) for w in tf)
            if np.isnan(score):
                score = 0
            text.score = score
        return texts
