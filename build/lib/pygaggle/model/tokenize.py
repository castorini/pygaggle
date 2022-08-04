from dataclasses import dataclass
from functools import lru_cache
from typing import List, Mapping, Union, Iterable, Optional, Tuple

from spacy.lang.en import English
from transformers import PreTrainedTokenizer
import torch

from pygaggle.rerank.base import Query, Text, TextType


__all__ = ['BatchTokenizer',
           'T5BatchTokenizer',
           'T5DuoBatchTokenizer',
           'QueryDocumentBatch',
           'DuoQueryDocumentBatch',
           'SimpleBatchTokenizer',
           'QueryDocumentBatchTokenizer',
           'SpacySenticizer',
           'SpacyWordTokenizer']
TokenizerReturnType = Mapping[str, Union[torch.Tensor, List[int],
                                         List[List[int]],
                                         List[List[str]]]]


@dataclass
class TokenizerOutputBatch:
    output: TokenizerReturnType
    texts: List[TextType]

    def __len__(self):
        return len(self.texts)


@dataclass
class QueryDocumentBatch:
    query: Query
    documents: List[Text]
    output: Optional[TokenizerReturnType] = None

    def __len__(self):
        return len(self.documents)


@dataclass
class DuoQueryDocumentBatch:
    query: Query
    doc_pairs: List[Tuple[Text, Text]]
    output: Optional[TokenizerReturnType] = None

    def __len__(self):
        return len(self.doc_pairs)


class TokenizerEncodeMixin:
    tokenizer: PreTrainedTokenizer = None
    tokenizer_kwargs = None

    def encode(self, strings: List[str]) -> TokenizerReturnType:
        assert self.tokenizer and self.tokenizer_kwargs is not None, \
                'mixin used improperly'
        ret = self.tokenizer.batch_encode_plus(strings,
                                               **self.tokenizer_kwargs)
        ret['tokens'] = list(map(self.tokenizer.tokenize, strings))
        return ret


class BatchTokenizer(TokenizerEncodeMixin):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 batch_size: int,
                 **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tokenizer_kwargs = tokenizer_kwargs

    def traverse(
            self,
            batch_input: List[TextType]) -> Iterable[TokenizerOutputBatch]:
        for batch_idx in range(0, len(batch_input), self.batch_size):
            inputs = batch_input[batch_idx:batch_idx + self.batch_size]
            input_ids = self.encode([x.text for x in inputs])
            yield TokenizerOutputBatch(input_ids, inputs)


class QueryDocumentBatchTokenizer(TokenizerEncodeMixin):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 batch_size: int,
                 pattern: str = '{query} {document}',
                 **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tokenizer_kwargs = tokenizer_kwargs
        self.pattern = pattern

    def traverse_query_document(
            self,
            batch_input: QueryDocumentBatch) -> Iterable[QueryDocumentBatch]:
        query = batch_input.query
        for batch_idx in range(0, len(batch_input), self.batch_size):
            docs = batch_input.documents[batch_idx:batch_idx + self.batch_size]
            outputs = self.encode([self.pattern.format(
                                        query=query.text,
                                        document=doc.text) for doc in docs])
            yield QueryDocumentBatch(query, docs, outputs)

    def traverse_duo_query_document(
            self,
            batch_input: DuoQueryDocumentBatch) -> Iterable[DuoQueryDocumentBatch]:
        query = batch_input.query
        for batch_idx in range(0, len(batch_input), self.batch_size):
            docs = batch_input.doc_pairs[batch_idx:batch_idx + self.batch_size]
            outputs = self.encode([self.pattern.format(
                                        query=query.text,
                                        document0=doc[0].text,
                                        document1=doc[1].text) for doc in docs])
            yield DuoQueryDocumentBatch(query, docs, outputs)


class T5BatchTokenizer(QueryDocumentBatchTokenizer):
    def __init__(self, *args, **kwargs):
        kwargs['pattern'] = 'Query: {query} Document: {document} Relevant:'
        if 'return_attention_mask' not in kwargs:
            kwargs['return_attention_mask'] = True
        if 'padding' not in kwargs:
            kwargs['padding'] = 'longest'
        if 'truncation' not in kwargs:
            kwargs['truncation'] = True
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'
        if 'max_length' not in kwargs:
            kwargs['max_length'] = 512
        super().__init__(*args, **kwargs)


class T5DuoBatchTokenizer(QueryDocumentBatchTokenizer):
    def __init__(self, *args, **kwargs):
        kwargs['pattern'] = 'Query: {query} Document0: {document0} Document1: {document1} Relevant:'
        if 'return_attention_mask' not in kwargs:
            kwargs['return_attention_mask'] = True
        if 'padding' not in kwargs:
            kwargs['padding'] = 'longest'
        if 'truncation' not in kwargs:
            kwargs['truncation'] = True
        if 'return_tensors' not in kwargs:
            kwargs['return_tensors'] = 'pt'
        if 'max_length' not in kwargs:
            kwargs['max_length'] = 512
        super().__init__(*args, **kwargs)


class SimpleBatchTokenizer(BatchTokenizer):
    def __init__(self, *args, **kwargs):
        if 'return_attention_mask' not in kwargs:
            kwargs['return_attention_mask'] = True
        if 'padding' not in kwargs:
            kwargs['padding'] = 'longest'
        if 'truncation' not in kwargs:
            kwargs['truncation'] = True
        super().__init__(*args, **kwargs)


class SpacyWordTokenizer:
    nlp = English()
    tokenizer = nlp.tokenizer

    @lru_cache(maxsize=1024)
    def __call__(self, text: str) -> List[str]:
        return list(x.text for x in self.tokenizer(text))


class SpacySenticizer:
    nlp = English()
    nlp.add_pipe('sentencizer')

    def __init__(self, max_paragraph_length: int = None):
        self.max_paragraph_length = max_paragraph_length
           
    @lru_cache(maxsize=1024)
    def __call__(self, document: str) -> List[str]:
        return [s.text for s in self.nlp(
            document[:self.max_paragraph_length]).sents]
            
