from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
from typing import List
import json
import re

from pyserini.search import pysearch

from pygaggle.rerank import Query, Text


__all__ = ['RelevanceExample', 'LuceneDocumentLoader']


@dataclass
class RelevanceExample:
    query: Query
    documents: List[Text]
    labels: List[bool]


class LuceneDocumentLoader:
    double_space_pattern = re.compile(r'\s\s+')

    def __init__(self, index_path: str):
        self.searcher = pysearch.SimpleSearcher(index_path)

    @lru_cache(maxsize=1024)
    def load_document(self, id: str) -> str:
        try:
            article = json.loads(self.searcher.doc(id).lucene_document().get('raw'))
        except json.decoder.JSONDecodeError:
            raise ValueError('article not found')
        except AttributeError:
            raise ValueError('document unretrievable')
        ref_entries = article['ref_entries'].values()
        text = '\n'.join(x['text'] for x in chain(article['abstract'], article['body_text'], ref_entries))
        return text
