from typing import List, Optional, Dict

from transformers import DPRReader, DPRReaderTokenizer

from pygaggle.qa.base import Reader, Answer, Question, Context
from pygaggle.qa.span_selection import DprSelection


class DprReader(Reader):
    """Class containing the DPR Reader
    Takes in a question and a list of the top passages selected by the retrieval model,
    and predicts a list of the best answer spans from the most relevant passages.

    Parameters
    ----------
    model_name : DPR Reader model name or path
    tokenizer_name : DPR Reader tokenizer name or path
    num_spans : Number of answer spans to return
    max_answer_length : Maximum length that an answer span can be
    num_spans_per_passage : Maximum number of answer spans to return per passage
    """

    def __init__(
            self,
            model_name: str,
            tokenizer_name: str = None,
            span_selection_rules=None,
            num_spans: int = 1,
            max_answer_length: int = 10,
            num_spans_per_passage: int = 10,
            batch_size: int = 16,
            device: str = 'cuda:0'
    ):
        if span_selection_rules is None:
            span_selection_rules = [DprSelection()]
        self.device = device
        self.model = DPRReader.from_pretrained(model_name).to(self.device).eval()
        if tokenizer_name:
            self.tokenizer = DPRReaderTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = DPRReaderTokenizer.from_pretrained(model_name)
        self.span_selection_rules = span_selection_rules
        self.num_spans = num_spans
        self.max_answer_length = max_answer_length
        self.num_spans_per_passage = num_spans_per_passage
        self.batch_size = batch_size

    def compute_spans(
            self,
            question: Question,
            contexts: List[Context],
    ):
        spans_for_contexts = []

        for i in range(0, len(contexts), self.batch_size):
            encoded_inputs = self.tokenizer(
                questions=question.text,
                titles=list(map(lambda t: t.title, contexts[i: i + self.batch_size])),
                texts=list(map(lambda t: t.text, contexts[i: i + self.batch_size])),
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

            # collect spans for each context
            batch_spans_by_contexts = [[] for _ in range(len(outputs.relevance_logits))]
            for span in predicted_spans:
                batch_spans_by_contexts[span.doc_id].append(span)

            # sort spans by span score
            for k in range(len(outputs.relevance_logits)):
                spans_for_contexts.append(
                    sorted(batch_spans_by_contexts[k], reverse=True, key=lambda span: span.span_score)
                )

        return spans_for_contexts

    def predict(
            self,
            question: Question,
            contexts: List[Context],
            topk_retrievals: Optional[List[int]] = None,
    ) -> Dict[int, List[Answer]]:
        if isinstance(question, str):
            question = Question(question)
        if topk_retrievals is None:
            topk_retrievals = [len(contexts)]

        answers = {str(rule): {} for rule in self.span_selection_rules}
        prev_topk_retrieval = 0
        for rule in self.span_selection_rules:
            rule.reset()

        spans = self.compute_spans(question, contexts)

        for topk_retrieval in topk_retrievals:
            for rule in self.span_selection_rules:
                rule.add_answers(
                    spans[prev_topk_retrieval: topk_retrieval],
                    contexts[prev_topk_retrieval: topk_retrieval]
                )
                answers[str(rule)][topk_retrieval] = rule.top_answers(self.num_spans)

            prev_topk_retrieval = topk_retrieval

        return answers
