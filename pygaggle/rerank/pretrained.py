import torch

from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          T5ForConditionalGeneration)

from .base import Reranker
from .transformer import (SequenceClassificationTransformerReranker,
                          T5Reranker)
from pygaggle.model import T5BatchTokenizer


__all__ = ['monoT5',
           'monoBERT']


def monoT5(model_name: str = 'castorini/monot5-base-msmarco',
           tokenizer_name: str = 't5-base',
           batch_size: int = 8) -> Reranker:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer = T5BatchTokenizer(tokenizer, batch_size)
    return T5Reranker(model, tokenizer)


def monoBERT(model_name: str = 'castorini/monobert-large-msmarco',
             tokenizer_name: str = 'bert-large-uncased') -> Reranker:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return SequenceClassificationTransformerReranker(model, tokenizer)
