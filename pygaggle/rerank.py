from typing import List

from pyserini.pyclass import JSimpleSearcherResult


class Query:
    """Class representing a query. A query contains the query text itself and potentially other metadata.

    Parameters
    ----------
    text : str
        The query text.
    """
    def __init__(self, text: str):
        self.text = text


class Text:
    """Class representing a text to be reranked. A text is unspecified with respect to it length; in principle, it
    could be a full-length document, a paragraph-sized passage, or even a short phrase.

    Parameters
    ----------
    contents : str
        The text to be reranked.
    raw : str
        The raw representation of the text to be ranked. For example, the ```raw``` might be a JSON object containing
        the ```contents``` as well as additional metadata data and other annotations.
    score : float
        The score of the text. For example, the score might be the BM25 score from an initial retrieval stage.
    """

    def __init__(self, contents: str, raw: str, score: float):
        self.contents = contents
        self.raw = raw
        self.score = score


class Reranker:
    """Class representing a reranker. A reranker takes a list texts and returns a list of texts non-destructively
    (i.e., does not alter the original input list of texts).
    """

    def __init__(self):
        pass

    def rerank(self, query: Query, texts: List[Text]):
        """Reranks a list of texts with respect to a query.

         Parameters
         ----------
         query : Query
             The query.
         texts : List[Text]
             The list of texts.

         Returns
         -------
         List[Text]
             Reranked list of texts.
         """
        pass


class IdentityReranker(Reranker):
    """A reranker that simply returns a clone of the input list of texts.
    """

    def rerank(self, query: Query, texts: List[Text]):
        output = []
        for text in texts:
            output.append(Text(text.contents, text.raw, text.score))
        return output


def to_texts(hits: List[JSimpleSearcherResult]):
    """Converts hits from Pyserini into a list of texts.

     Parameters
     ----------
     hits : List[JSimpleSearcherResult]
         The hits.

     Returns
     -------
     List[Text]
         List of texts.
     """
    texts = []
    for i in range(0, len(hits)):
        texts.append(Text(hits[i].contents, hits[i].raw, hits[i].score))
    return texts
