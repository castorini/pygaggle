from copy import deepcopy
from typing import List
import random

from . import Query, Text
from .base import Reranker


__all__ = ['RandomReranker']


class RandomReranker(Reranker):
    def __init__(self, seed: int = 0):
        self.rand = random.Random(seed)

    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        for text in texts:
            text.score = self.rand.random()
        return texts
