from collections import OrderedDict
from typing import List
import json
import logging

from pydantic import BaseModel

from .relevance import RelevanceExample, LuceneDocumentLoader
from pygaggle.model.tokenize import SpacySenticizer
from pygaggle.rerank import Query, Text


__all__ = ['MISSING_ID', 'LitReviewCategory', 'LitReviewAnswer', 'LitReviewDataset', 'LitReviewSubcategory']


MISSING_ID = '<missing>'


class LitReviewAnswer(BaseModel):
    id: str
    title: str
    exact_answer: str


class LitReviewSubcategory(BaseModel):
    name: str
    answers: List[LitReviewAnswer]


class LitReviewCategory(BaseModel):
    name: str
    sub_categories: List[LitReviewSubcategory]


class LitReviewDataset(BaseModel):
    categories: List[LitReviewCategory]

    @classmethod
    def from_file(cls, filename: str) -> 'LitReviewDataset':
        with open(filename) as f:
            return cls(**json.load(f))

    @property
    def query_answer_pairs(self):
        return ((subcat.name, ans) for cat in self.categories
                for subcat in cat.sub_categories
                for ans in subcat.answers)

    def to_senticized_dataset(self, index_path: str) -> List[RelevanceExample]:
        loader = LuceneDocumentLoader(index_path)
        tokenizer = SpacySenticizer()
        example_map = OrderedDict()
        rel_map = OrderedDict()
        for query, document in self.query_answer_pairs:
            if document.id == MISSING_ID:
                logging.warning(f'Skipping {document.title} (missing ID)')
                continue
            key = (query, document.id)
            example_map.setdefault(key, tokenizer(loader.load_document(document.id)))
            sents = example_map[key]
            rel_map.setdefault(key, [False] * len(sents))
            for idx, s in enumerate(sents):
                if document.exact_answer in s:
                    rel_map[key][idx] = True
        for (_, doc_id), rels in rel_map.items():
            if not any(rels):
                logging.warning(f'{doc_id} has no relevant answers')
        return [RelevanceExample(Query(query), list(map(Text, sents)), rels)
                for ((query, _), sents), (_, rels) in zip(example_map.items(), rel_map.items())]
