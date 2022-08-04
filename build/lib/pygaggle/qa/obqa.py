import json
from .base import Context
import time


class OpenBookQA:

    def __init__(self, reader, retriever, corpus):
        self.reader = reader
        self.retriever = retriever
        self.corpus = corpus

    def predict(self, question, topk=20, query='', reader_name='dpr'):
        hits = self.retriever.search(query + question, topk)
        contexts = self._hits_to_contexts(hits)
        answer = self.reader.predict(question, contexts)
        if reader_name == 'fid':
            return answer 
        answer = answer[str(self.reader.span_selection_rules[0])][topk][0]
        return self._parse_answer(answer)

    def _hits_to_contexts(self, hits, title_delimiter='\n'):
        """
            Converts hits from Pyserini into a list of contexts.
        """
        contexts = []
        for i in range(0, len(hits)):
            docid = str(hits[i].docid)
            t = json.loads(self.corpus.doc(docid).raw())['contents']
            if title_delimiter:
                title, t = t.split(title_delimiter)
                contexts.append(Context(t, title, docid, hits[i].score))
            else:
                contexts.append(Context(t, None, docid, hits[i].score))
        return contexts

    @staticmethod
    def _parse_answer(answer):
        return {"answer": answer.text,
                "context": {
                    "docid": answer.context.docid,
                    "title": answer.context.title,
                    "text": answer.context.text}
                }
