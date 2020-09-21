import spacy
import numpy as np
from pygaggle.rerank.base import Text
from typing import List
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class SegmentGroup:
    """
    'segments' stores the List of document segments
    'doc_end_indexes' stores the index of the last segment of each
        document when 'segment()' converting a 'List[Text]' of documents into
        'List[Text]' of segments. It will be used to split and group segments'
        scores and feed the aggregated scores back to documents in 'aggregate()'
    """
    segments: List[Text]
    doc_end_indexes: List[int]


class SegmentProcessor:

    def __init__(self, max_characters=10000000):
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))
        self.max_characters = max_characters
        self.aggregate_methods = {
            "max": self._max_aggregate,
            "mean": self._mean_aggregate
        }

    def segment(self, documents: List[Text], seg_size: int, stride: int) -> SegmentGroup:
        segmented_doc, doc_end_indexes, end_idx = [], [0], 0
        for document in documents:
            doc = self.nlp(document.text[:self.max_characters])
            sentences = [sent.string.strip() for sent in doc.sents]
            if len(sentences) == 0: # Text is empty
                segment_text = document.title + '. '
                segmented_doc.append(Text(segment_text, dict(docid=document.metadata["docid"])))
                end_idx += 1
                doc_end_indexes.append(int(end_idx))
            else:
                for i in range(0, len(sentences), stride):
                    segment_text = ' '.join(sentences[i:i + seg_size])
                    segment_text = document.title + '. ' + segment_text
                    segmented_doc.append(Text(segment_text, dict(docid=document.metadata["docid"])))
                    if i + seg_size >= len(sentences):
                        end_idx += i/stride + 1
                        doc_end_indexes.append(int(end_idx))
                        break
        return SegmentGroup(segmented_doc, doc_end_indexes)

    def aggregate(self, documents: List[Text], segments_group: SegmentGroup, method: str = "max") -> List[Text]:
        docs = deepcopy(documents)
        for i in range(len(docs)):
            doc_start_idx = segments_group.doc_end_indexes[i]
            doc_end_idx = segments_group.doc_end_indexes[i+1]
            target_scores = [seg.score for seg in segments_group.segments[doc_start_idx: doc_end_idx]]
            docs[i].score = self.aggregate_methods[method](target_scores)
        return docs

    @staticmethod
    def _max_aggregate(scores):
        return max(scores)

    @staticmethod
    def _mean_aggregate(scores):
        return np.mean(scores)
