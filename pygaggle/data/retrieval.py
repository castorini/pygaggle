from dataclasses import dataclass
from typing import List

from pygaggle.qa.base import Question, Context


@dataclass
class RetrievalExample:
    question: Question
    contexts: List[Context]
    ground_truth_answers: List[List[str]]
