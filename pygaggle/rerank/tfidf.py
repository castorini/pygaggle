from collections import Counter
from copy import deepcopy
from typing import Callable, List
import math

import numpy as np

from pygaggle.rerank import Reranker, Query, Text


__all__ = ['TfIdfReranker']


class TfIdfReranker(Reranker):
    def __init__(self,
                 word_tokenizer: Callable[[str], List[str]],
                 k1: float = 1.6,
                 b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.word_tokenizer = word_tokenizer

    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        query_words = self.word_tokenizer(query.text)
        sentences = list(map(self.word_tokenizer, (t.text for t in texts)))

        query_words_lower = {x.lower() for x in query_words}
        sentences_lower = [[w.lower() for w in sent] for sent in sentences]
        sentence_sets = list(map(set, sentences_lower))
        idfs = {w: math.log(len(sentence_sets) / (1 + sum(int(w in sent) for sent in sentence_sets)))
                for w in query_words_lower}
        mean_len = np.mean(list(map(len, sentences)))
        d_len = len(sentences)
        texts = deepcopy(texts)
        for sent_words, text in zip(sentences_lower, texts):
            tf = Counter(filter(query_words_lower.__contains__, sent_words))
            score = sum(idfs[w] * tf[w] * (self.k1 + 1) /
                        (tf[w] + self.k1 * (1 - self.b + self.b * (d_len / mean_len))) for w in tf)
            text.score = score
        return texts
