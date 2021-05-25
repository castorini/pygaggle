from typing import List, Optional
import abc

__all__ = ['Question', 'Answer', 'Context', 'Reader']


class Question:
    """Class representing a question.
    A question contains the question text itself and potentially other metadata.

    Parameters
    ----------
    text : str
        The question text.
    qid : Optional[str]
        The question id.
    """
    def __init__(self, text: str, qid: Optional[str] = None):
        self.text = text
        self.id = qid


class Context:
    """Class representing a context to exact answer from.
    A text is unspecified with respect to it length; in principle, it
    could be a full-length document, a paragraph-sized passage, or
    even a short phrase.

    Parameters
    ----------
    text : str
        The context to extract answer from.
    score : Optional[float]
        The score of the context. For example, the score might be the BM25 score
        from an initial retrieval stage.
    title : Optional[str]
        The context's title.
    """

    def __init__(self,
                 text: str,
                 title: Optional[str] = None,
                 docid: Optional[str] = None,
                 score: Optional[float] = 0):
        self.text = text
        self.title = title
        self.docid = docid
        self.score = score


class Answer:
    """
    Class representing an answer.
    An answer contains the answer text itself and potentially other metadata.
    Parameters
    ----------
    text : str
        The answer text.
    score : Optional[float]
        The score of the answer.
    """
    def __init__(
            self,
            text: str,
            context: Optional[Context] = None,
            score: Optional[float] = 0.0,
    ):
        self.text = text
        self.context = context
        self.score = score


class Reader:
    """
    Class representing a Reader.
    A Reader takes a Query and a list of Text and returns a list of Answer.
    """
    @abc.abstractmethod
    def predict(
        self,
        question: Question,
        contexts: List[Context],
    ) -> List[Answer]:
        """
        Find answers from a list of texts with respect to a question.
        Parameters
        ----------
        question : Question
            The question.
        contexts : List[Context]
            The list of contexts.
        Returns
        -------
        List[Answer]
            Predicted list of answers.
        """
        pass
