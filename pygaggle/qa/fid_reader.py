from pygaggle.qa.base import Reader, Answer, Question, Context
from pygaggle.qa.span_selection import DprSelection
from typing import List, Optional, Dict
import transformers 

import torch
from torch.utils.data import DataLoader, SequentialSampler

from pygaggle.qa.base import Reader, Answer, Question, Context
import pygaggle.qa.fid.model as fid_model

class FidReader(Reader):
    def __init__(
            self,
            model_name: str,
            tokenizer_name: str = 't5-base',
            span_selection_rules=None,
            num_spans: int = 1,
            max_answer_length: int = 10,
            num_spans_per_passage: int = 10,
            text_maxlength: int = 250,
            batch_size: int = 16,
            device: str = 'cuda:0'
    ):
        if span_selection_rules is None:
            span_selection_rules = [DprSelection()]
        self.device = device

        model_class = fid_model.FiDT5
        self.model = model_class.from_pretrained('../FiD/pretrained_models/' + model_name).to(self.device).eval()
        self.tokenizer =  transformers.T5Tokenizer.from_pretrained(tokenizer_name, return_dict=False)
        # FiD uses generate to compute spans

        self.span_selection_rules = span_selection_rules
        self.num_spans = num_spans
        self.max_answer_length = max_answer_length
        self.num_spans_per_passage = num_spans_per_passage
        self.text_maxlength = text_maxlength
        self.batch_size = batch_size

    # expanded upon Collator used by DataLoader
    def build_dataloader(
            self,
            question: Question,
            contexts: List[Context], #corresponde to passages in fid
    ):
        for i in range(0, len(contexts)):
            contexts[i].text = "question: " + question.text + " " + \
                                "title: " + contexts[i].title + " " + \
                                "context: " + contexts[i].text

        p = self.tokenizer.batch_encode_plus(
                [c.text for c in contexts],
                max_length=self.text_maxlength,
                pad_to_max_length=True,
                return_tensors='pt',
                truncation=True
            )
        return (p['input_ids'][None], p['attention_mask'][None].bool())   


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
        batch = self.build_dataloader(question, contexts)
        answers = {str(rule): {} for rule in self.span_selection_rules}
        with torch.no_grad():
            (context_ids, context_mask) = batch
            outputs = self.model.generate(
                input_ids=context_ids.to(self.device),
                attention_mask=context_mask.to(self.device),
                max_length=350 #from dpr_reader
            )
            for k, o in enumerate(outputs):
                ans = self.tokenizer.decode(o, skip_special_tokens=True)
        return ans
