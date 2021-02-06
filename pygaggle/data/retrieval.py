from dataclasses import dataclass
from typing import List

from pygaggle.rerank.base import Query, Text


@dataclass
class RetrievalExample:
    query: Query
    texts: List[Text]
    ground_truth_answers: List[List[str]]
