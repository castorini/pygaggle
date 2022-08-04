from typing import List, Union, Optional, Mapping, Any
from copy import deepcopy
import abc

from pyserini.search import JLuceneSearcherResult


__all__ = ['Query', 'Text', 'Reranker', 'hits_to_texts', 'TextType']


TextType = Union['Query', 'Text']


class Query:
    """Class representing a query.
    A query contains the query text itself and potentially other metadata.

    Parameters
    ----------
    text : str
        The query text.
    id : Optional[str]
        The query id.
    """

    def __init__(self, text: str, id: Optional[str] = None):
        self.text = text
        self.id = id


class Text:
    """Class representing a text to be reranked.
    A text is unspecified with respect to it length; in principle, it
    could be a full-length document, a paragraph-sized passage, or
    even a short phrase.

    Parameters
    ----------
    text : str
        The text to be reranked.
    metadata : Mapping[str, Any]
        Additional metadata and other annotations.
    score : Optional[float]
        The score of the text. For example, the score might be the BM25 score
        from an initial retrieval stage.
    title : Optional[str]
        The text's title.
    """

    def __init__(self,
                 text: str,
                 metadata: Mapping[str, Any] = None,
                 score: Optional[float] = 0,
                 title: Optional[str] = None):
        self.text = text
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.score = score
        self.title = title


class Reranker:
    """Class representing a reranker.
    A reranker takes a list texts and returns a list of texts non-destructively
    (i.e., does not alter the original input list of texts).
    """

    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        """Sorts a list of texts
        """
        return sorted(self.rescore(query, texts), key=lambda x: x.score, reverse=True)

    @abc.abstractmethod
    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
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


def hits_to_texts(hits: List[JLuceneSearcherResult], field='raw') -> List[Text]:
    """Converts hits from Pyserini into a list of texts.

     Parameters
     ----------
     hits : List[JLuceneSearcherResult]
        The hits.
     field : str
        Field to use.

     Returns
     -------
     List[Text]
         List of texts.
     """
    texts = []
    for i in range(0, len(hits)):
        t = hits[i].raw if field == 'raw' else hits[i].contents
        metadata = {'raw': hits[i].raw, 'docid': hits[i].docid}
        texts.append(Text(t, metadata, hits[i].score))
    return texts
