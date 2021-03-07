from typing import List, Optional, Mapping, Any
import abc

from pygaggle.rerank.base import Query, Text

__all__ = ['Answer', 'Reader']


class Answer:
    """
    Class representing an answer.
    An answer contains the answer text itself and potentially other metadata.
    Parameters
    ----------
    text : str
        The answer text.
    metadata : Mapping[str, Any]
        Additional metadata and other annotations.
    language: str
        The language of the answer text.
    score : Optional[float]
        The score of the answer.
    ctx_score : Optional[float]
        The context score of the answer.
    total_score : Optional[float]
        The aggregated score of answer score and ctx_score
    """
    def __init__(
        self,
        text: str,
        language: str = "en",
        metadata: Mapping[str, Any] = None,
        score: Optional[float] = 0,
        ctx_score: Optional[float] = 0,
        total_score: Optional[float] = 0,
    ):
        self.text = text
        self.language = language
        if metadata is None:
            metadata = dict()
        self.metadata = metadata
        self.score = score
        self.ctx_score = ctx_score
        self.total_score = total_score

    def aggregate_score(self, weight):
        self.total_score = weight*self.score + (1-weight)*self.ctx_score


class Reader:
    """
    Class representing a Reader.
    A Reader takes a Query and a list of Text and returns a list of Answer.
    """
    @abc.abstractmethod
    def predict(
        self,
        query: Query,
        texts: List[Text],
    ) -> List[Answer]:
        """
        Find answers from a list of texts with respect to a query.
        Parameters
        ----------
        query : Query
            The query.
        texts : List[Text]
            The list of texts.
        Returns
        -------
        List[Answer]
            Predicted list of answers.
        """
        pass
