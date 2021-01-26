from typing import List

from transformers import DPRReader, DPRReaderTokenizer

from .base import Reader, Answer
from ..reranker.base import Query, Text

class DPRReader(Reader):
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str = None,
        num_spans: int = 16,
        max_answer_length: int = 64,
        num_spans_per_passage: int = 4,
    ):
        if tokenizer_name is None:
            tokenizer_name = model_name

        self.tokenizer = DPRReaderTokenizer.from_pretrained(tokenizer_name)
        self.model = DPRReader.from_pretrained(model_name)

        self.num_spans = num_spans
        self.max_answer_length = max_answer_length
        self.num_spans_per_passage = num_spans_per_passage

    def predict(
        self,
        query: Query,
        texts: List[Text],
    ) -> List[Answer]:
        # Each text should have format "title\ncontent"
        titles, contents = tuple(map(list, zip(*map(lambda t: t.text.split('\n', 1), texts))))

        encoded_inputs = self.tokenizer(
            questions=[query.text],
            titles=titles,
            texts=contents,
            return_tensors='pt',
        )
        outputs = self.model(**encoded_inputs)

        predicted_spans = tokenizer.decode_best_spans(
            encoded_inputs,
            outputs,
            self.num_spans,
            self.max_answer_length,
            self.num_spans_per_passage,
        )

        answers = []
        for idx, span in enumerate(predicted_spans):
            answers.append(
                Answer(
                    text=span.text,
                    score=span.span_score,
                    ctx_score=span.relevance_score,
                )
            )

        return answers
