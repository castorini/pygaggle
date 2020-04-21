from copy import deepcopy
from typing import List

from transformers import T5ForConditionalGeneration, PreTrainedModel
import torch

from pygaggle.model import greedy_decode, QueryDocumentBatchTokenizer, BatchTokenizer,\
    QueryDocumentBatch, LongBatchEncoder, SpecialTokensCleaner
from pygaggle.rerank import Reranker, Query, Text, SimilarityMatrixProvider


__all__ = ['T5Reranker', 'TransformerReranker']


class T5Reranker(Reranker):
    def __init__(self, model: T5ForConditionalGeneration, tokenizer: QueryDocumentBatchTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(self.model.parameters(), None).device

    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        batch_input = QueryDocumentBatch(query=query, documents=texts)
        for batch in self.tokenizer.traverse_query_document(batch_input):
            input_ids = batch.output['input_ids']
            attn_mask = batch.output['attention_mask']
            _, batch_scores = greedy_decode(self.model,
                                            input_ids.to(self.device),
                                            length=2,
                                            attention_mask=attn_mask.to(self.device),
                                            return_last_logits=True)

            # 6136 and 1176 are the indexes of the tokens false and true in T5.
            batch_scores = batch_scores[:, [6136, 1176]]
            batch_log_probs = torch.nn.functional.log_softmax(batch_scores, dim=1)
            batch_log_probs = batch_log_probs[:, 1].tolist()
            for doc, score in zip(batch.documents, batch_log_probs):
                doc.score = score
        return texts


class TransformerReranker(Reranker):
    methods = dict(max=lambda x: x.max().item(),
                   mean=lambda x: x.mean().item(),
                   absmean=lambda x: x.abs().mean().item(),
                   absmax=lambda x: x.abs().max().item())

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: BatchTokenizer,
                 sim_matrix_provider: SimilarityMatrixProvider,
                 method: str = 'max',
                 clean_special: bool = True):
        assert method in self.methods, 'inappropriate scoring method'
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = LongBatchEncoder(model, tokenizer)
        self.sim_matrix_provider = sim_matrix_provider
        self.method = method
        self.clean_special = clean_special
        self.cleaner = SpecialTokensCleaner(tokenizer.tokenizer)
        self.device = next(self.model.parameters(), None).device

    @torch.no_grad()
    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        encoded_query = self.encoder.encode_single(query)
        encoded_documents = self.encoder.encode(texts)
        texts = deepcopy(texts)
        for enc_doc, text in zip(encoded_documents, texts):
            if self.clean_special:
                enc_doc = self.cleaner.clean(enc_doc)
            matrix = self.sim_matrix_provider.compute_matrix(encoded_query, enc_doc)
            score = self.methods[self.method](matrix)
            text.score = score
        return texts
