from copy import deepcopy
from typing import List
import random

from .base import Query, Text, Reranker


__all__ = ['RandomReranker']


class RandomReranker(Reranker):
    def __init__(self, seed: int = 0):
        self.rand = random.Random(seed)

    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        for text in texts:
            text.score = self.rand.random()
        return texts
