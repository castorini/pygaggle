import spacy
from pygaggle.rerank.base import Text
from typing import List


class SegmentProcessor:

    def __init__(self, documents: List[Text]):
        self.doc_end_indexes = [0]
        self.documents = documents
        self.aggregate_methods = {
            "max": self._max_aggregate,
            "sum": self._sum_aggregate
        }

    def segment(self, seg_size: int, stride: int) -> List[Text]:
        segmented_doc, end_idx = [], 0
        nlp = spacy.blank("en")
        nlp.add_pipe(nlp.create_pipe("sentencizer"))
        for document in self.documents:
            doc = nlp(document.text[:1000000])
            sentences = [sent.string.strip() for sent in doc.sents]
            for i in range(0, len(sentences), stride):
                segment_text = ' '.join(sentences[i:i + seg_size])
                segmented_doc.append(Text(segment_text))
                if i + seg_size >= len(sentences):
                    end_idx += i/stride + 1
                    self.doc_end_index.append(int(end_idx))
                    break
        return segmented_doc

    def aggregate(self, scores: List[int], method: str = "max"):
        for i in range(len(self.documents)):
            scores_start_idx = self.doc_end_indexes[i]
            scores_end_idx = self.doc_end_indexes[i+1]
            target_scores = scores[scores_start_idx: scores_end_idx]
            self.documents[i].score = self.aggregate_methods[method](target_scores)
        return self.documents

    @staticmethod
    def _max_aggregate(scores):
        return max(scores)

    @staticmethod
    def _sum_aggregate(scores):
        return sum(scores)
