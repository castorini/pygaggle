from typing import List, Optional, Dict
import torch

from transformers import DPRReader, DPRReaderTokenizer

from .base import Reader, Answer
from .reader_settings import DPRBaseSettings, DPRSettings
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
        reader_settings: List[DPRBaseSettings] = [DPRSettings()],
        num_spans: int = 1,
        max_answer_length: int = 10,
        num_spans_per_passage: int = 10,
        batch_size: int = 16,
    ):
        self.model = model or self.get_model()
        self.tokenizer = tokenizer or self.get_tokenizer()
        self.device = next(self.model.parameters(), None).device

        self.reader_settings = reader_settings

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

    def compute_spans(
        self,
        query: Query,
        texts: List[Text],
    ):
        spans_by_text = []

        for i in range(0, len(texts), self.batch_size):
            encoded_inputs = self.tokenizer(
                questions=query.text,
                titles=list(map(lambda t: t.title, texts[i: i+self.batch_size])),
                texts=list(map(lambda t: t.text, texts[i: i+self.batch_size])),
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=350,
            )
            input_ids = encoded_inputs['input_ids'].to(self.device)
            attention_mask = encoded_inputs['attention_mask'].to(self.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            outputs.start_logits = outputs.start_logits.cpu().detach().numpy()
            outputs.end_logits = outputs.end_logits.cpu().detach().numpy()
            outputs.relevance_logits = outputs.relevance_logits.cpu().detach().numpy()

            predicted_spans = self.tokenizer.decode_best_spans(
                encoded_inputs,
                outputs,
                self.batch_size * self.num_spans_per_passage,
                self.max_answer_length,
                self.num_spans_per_passage,
            )

            batch_spans_by_text = [[] for k in range(len(outputs.relevance_logits))]
            for span in predicted_spans:
                batch_spans_by_text[span.doc_id].append(span)

            for k in range(len(outputs.relevance_logits)):
                spans_by_text.append(sorted(batch_spans_by_text[k], reverse=True, key=lambda span: span.span_score))

        return spans_by_text

    def predict(
        self,
        query: Query,
        texts: List[Text],
        milestones: Optional[List[int]] = None,
    ) -> Dict[int, List[Answer]]:
        if milestones is None:
            milestones = [len(texts)]

        answers = {str(setting): {} for setting in self.reader_settings}
        top_answers = []
        prev_milestone = 0
        for setting in self.reader_settings:
            setting.reset()

        spans_by_text = self.compute_spans(query, texts)

        for milestone in milestones:
            for setting in self.reader_settings:
                setting.add_answers(spans_by_text[prev_milestone: milestone], texts[prev_milestone: milestone])
                answers[str(setting)][milestone] = setting.top_answers(self.num_spans)

            prev_milestone = milestone

        return answers
