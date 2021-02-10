from typing import List

from .base import Query, Text, Reranker


__all__ = ['IdentityReranker']


class IdentityReranker(Reranker):
    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        return texts
