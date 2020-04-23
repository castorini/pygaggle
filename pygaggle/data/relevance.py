from dataclasses import dataclass
from functools import lru_cache
from typing import List
import json
import re

from pyserini.search import pysearch

from pygaggle.rerank.base import Query, Text


__all__ = ['RelevanceExample', 'Cord19DocumentLoader']


@dataclass
class RelevanceExample:
    query: Query
    documents: List[Text]
    labels: List[bool]


@dataclass
class Cord19Document:
    abstract: str
    body_text: str
    ref_entries: str

    @property
    def all_text(self):
        return '\n'.join((self.abstract, self.body_text, self.ref_entries))


class Cord19DocumentLoader:
    double_space_pattern = re.compile(r'\s\s+')

    def __init__(self, index_path: str):
        self.searcher = pysearch.SimpleSearcher(index_path)

    @lru_cache(maxsize=1024)
    def load_document(self, id: str) -> Cord19Document:
        def unfold(entries):
            return '\n'.join(x['text'] for x in entries)
        try:
            article = json.loads(self.searcher.doc(id).lucene_document().get('raw'))
        except json.decoder.JSONDecodeError:
            raise ValueError('article not found')
        except AttributeError:
            raise ValueError('document unretrievable')
        ref_entries = article['ref_entries'].values()
        return Cord19Document(unfold(article['abstract']),
                              unfold(article['body_text']),
                              unfold(ref_entries))
