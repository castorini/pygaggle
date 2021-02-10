from typing import List, Optional
import torch

from transformers import DPRReader, DPRReaderTokenizer

from .base import Reader, Answer
from pygaggle.rerank.base import Query, Text


class DensePassageRetrieverReader(Reader):
    """Class containing the DPR Reader
    Takes in a query and a list of the top passages selected by the retrieval model,
    and predicts a list of the best answer spans from the most relevant passages, reranked.

    Parameters
    ----------
    model : DPR Reader model for predicting start, end, and relevance logits
    tokenizer : DPR Reader tokenizer for encoding input query and texts
    num_spans : Number of answer spans to return
    max_answer_length : Maximum length that an answer span can be
    num_spans_per_passage : Maximum number of answer spans to return per passage
    """

    def __init__(
        self,
        model: DPRReader = None,
        tokenizer: DPRReaderTokenizer = None,
        num_spans: int = 1,
        max_answer_length: int = 10,
        num_spans_per_passage: int = 10,
        batch_size: int = 16,
    ):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.device = next(self.model.parameters(), None).device

        self.num_spans = num_spans
        self.max_answer_length = max_answer_length
        self.num_spans_per_passage = num_spans_per_passage

        self.batch_size = batch_size

    @staticmethod
    def get_model(
        pretrained_model_name_or_path: str = 'facebook/dpr-reader-single-nq-base',
        device: Optional[str] = None,
    ) -> DPRReader:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return DPRReader.from_pretrained(pretrained_model_name_or_path).to(device).eval()

    @staticmethod
    def get_tokenizer(
        pretrained_tokenizer_name_or_path: str = 'facebook/dpr-reader-single-nq-base',
    ) -> DPRReaderTokenizer:
        return DPRReaderTokenizer.from_pretrained(pretrained_tokenizer_name_or_path)

    def predict(self, query: Query, texts: List[Text]) -> List[Answer]:
        answers = []
        for i in range(0, len(texts), self.batch_size):
            encoded_inputs = self.tokenizer(
                questions=query.text,
                titles=list(map(lambda t: t.title, texts[i: i+self.batch_size])),
                texts=list(map(lambda t: t.text, texts[i: i+self.batch_size])),
                return_tensors='pt',
                padding=True,
                truncation=True,
            )
            input_ids = encoded_inputs['input_ids'].to(self.device)
            attention_mask = encoded_inputs['attention_mask'].to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            predicted_spans = self.tokenizer.decode_best_spans(
                encoded_inputs,
                outputs,
                self.num_spans,
                self.max_answer_length,
                self.num_spans_per_passage,
            )

            for idx, span in enumerate(predicted_spans):
                answers.append(
                    Answer(
                        text=span.text,
                        score=float(span.span_score.cpu().detach().numpy()),
                        ctx_score=float(span.relevance_score.cpu().detach().numpy()),
                    )
                )
        answers = sorted(answers, key=lambda x: (-x.ctx_score, -x.score))
        return answers[: self.num_spans]
