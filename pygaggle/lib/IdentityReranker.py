from typing import List

from pygaggle.rerank import Reranker, Query, Text


class IdentityReranker(Reranker):
    """A reranker that simply returns a clone of the input list of texts.
    """

    def rerank(self, query: Query, texts: List[Text]):
        output = []
        for text in texts:
            output.append(Text(text.contents, text.raw, text.score))
        return output
