import sys
sys.path.insert(1, '/u0/o3liu/FiD')
# import test_reader
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
            tokenizer_name: str = None,
            span_selection_rules=None,
            num_spans: int = 1,
            max_answer_length: int = 10,
            num_spans_per_passage: int = 10,
            text_maxlength: int = 200,
            batch_size: int = 16,
            device: str = 'cuda:0'
    ):
        if span_selection_rules is None:
            span_selection_rules = [DprSelection()]
        self.device = device

        model_class = fid_model.FiDT5
        self.model = model_class.from_pretrained('/home/alvis/FiD/pretrained_models/nq_reader_base').to(self.device).eval()
        self.tokenizer =  transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)
        # FiD uses generate to compute spans

        self.span_selection_rules = span_selection_rules
        self.num_spans = num_spans
        self.max_answer_length = max_answer_length
        self.num_spans_per_passage = num_spans_per_passage
        self.batch_size = batch_size

    # expanded upon Collator used by DataLoader
    def build_dataloader(
            self,
            question: Question,
            contexts: List[Context], #corresponde to passages in fid
    ):
        # def encode_passages(batch_text_passages, tokenizer, max_length):
        #     passage_ids, passage_masks = [], []
        #     for k, text_passages in enumerate(batch_text_passages):
        #         p = tokenizer.batch_encode_plus(
        #             text_passages,
        #             max_length=max_length,
        #             pad_to_max_length=True,
        #             return_tensors='pt',
        #             truncation=True
        #         )
        #         passage_ids.append(p['input_ids'][None])
        #         passage_masks.append(p['attention_mask'][None])

        #     passage_ids = torch.cat(passage_ids, dim=0)
        #     passage_masks = torch.cat(passage_masks, dim=0)
        #     return passage_ids, passage_masks.bool()

        # def append_question(question, context):
        #     if contexts is None:
        #         return [question]
        #     return [Question(text=question.text + " " + t) for t in contexts]
        # text_passages = [append_question(question, context) for context in contexts]
        for i in range(0, len(contexts)):
            # if contexts[i] is None: # Just following fid, not sure if possible
            #     contexts[i] = [question]
            # else:
            contexts[i].text = "question: " + question.text + " " + \
                                "title: " + contexts[i].title + " " + \
                                "context: " + contexts[i].text
        # print(question.text)
        # print(contexts[0].text)
        # print(contexts[1].text)


        p = self.tokenizer.batch_encode_plus(
                [c.text for c in contexts],
                max_length=350,#self.text_maxlength,
                pad_to_max_length=True,
                return_tensors='pt',
                truncation=True
            )
        # print(p)

        return (p['input_ids'][None], p['attention_mask'][None].bool())
        
        # passage_ids, passage_masks = encode_passages(text_passages,
        #                                              self.tokenizer,
        #                                              self.text_maxlength)
        # batches= []
        # for i in range(0, len(contexts), self.batch_size):
        #     batch = contexts[i: i + self.batch_size]
        #     text_passages = [append_question(example) for example in batch]
        #     passage_ids, passage_masks = encode_passages(text_passages,
        #                                              self.tokenizer,
        #                                              self.text_maxlength)
        #     batches.append(passage_ids, passage_masks)
        # return batches

            
        


    def predict(
            self,
            question: Question,
            contexts: List[Context],
            topk_retrievals: Optional[List[int]] = None,
    ) -> Dict[int, List[Answer]]:
        batch = self.build_dataloader(question, contexts)
        answers = {str(rule): {} for rule in self.span_selection_rules}
        with torch.no_grad():
            (context_ids, context_mask) = batch
            # print(context_ids)
            # print(len(context_ids))
            # print(context_mask)
            outputs = self.model.generate(
                input_ids=context_ids.to(self.device),
                attention_mask=context_mask.to(self.device),
                max_length=350 #from dpr_reader
            )
            # print("outputs:")
            # print("outputs:", outputs)
            for k, o in enumerate(outputs):
                ans = self.tokenizer.decode(o, skip_special_tokens=True)
                print(ans)
            # print(ans)
        return ans
        return answers
            # for i, batch in enumerate(batches):
            #     (context_ids, context_mask) = batch
            #     outputs = self.model.generate(
            #         input_ids=context_ids.cuda(),
            #         attention_mask=context_mask.cuda(),
            #         max_length=50
            #     )
            #     for k, o in enumerate(outputs):
            #         ans = self.tokenizer.decode(o, skip_special_tokens=True)

        
if __name__ == "__main__":
    print("main function executed")
