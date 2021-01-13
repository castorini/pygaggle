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
    """
    The SegmentProcessor class is responsible for segmenting documents and aggregating the scores of segments from
    the same document.
    """
    def __init__(self, max_characters=10000000):
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))
        self.max_characters = max_characters
        self.aggregate_methods = {
            "max": self._max_aggregate,
            "mean": self._mean_aggregate
        }

    def segment(self, documents: List[Text], seg_size: int, stride: int) -> SegmentGroup:
        """
        Breaks each document into segments.  For example, given a document with sentences [1,2,3,4,5], a seg_size of 3,
        and a stride of 2, the document will be broken into segments [[1, 2, 3], [3, 4, 5], and [5]].  If the document's
        text is empty, a single segment containing the document's title is generated.  Otherwise, the document's title
        is prepended to the document's text.

        :param documents: A list of Text objects, each of which corresponds to an indexed document.
        :param seg_size: The number of sentences each segment should contain.
        :param stride: The number of sentences to advance for the next segment.
        :return: A SegmentGroup containing all the documents' segments and the end index of each document in
        segmented_docs.
        """
        segmented_docs, doc_end_indexes, end_idx = [], [0], 0
        for document in documents:
            doc = self.nlp(document.text[:self.max_characters])
            sentences = [sent.string.strip() for sent in doc.sents]
            # If the text is empty (i.e. there are no sentences), the segment_text is solely the title of the document.
            if len(sentences) == 0:
                segment_text = document.title
                segmented_docs.append(Text(segment_text, dict(docid=document.metadata["docid"])))
                end_idx += 1
                doc_end_indexes.append(int(end_idx))
            else:
                for i in range(0, len(sentences), stride):
                    segment_text = ' '.join(sentences[i:i + seg_size])
                    if document.title and (not document.title == ''):
                        segment_text = document.title + '. ' + segment_text
                    segmented_docs.append(Text(segment_text, dict(docid=document.metadata["docid"])))
                    if i + seg_size >= len(sentences):
                        end_idx += i/stride + 1
                        doc_end_indexes.append(int(end_idx))
                        break
        return SegmentGroup(segmented_docs, doc_end_indexes)

    def aggregate(self, documents: List[Text], segments_group: SegmentGroup, method: str = "max") -> List[Text]:
        """
        Aggregates the scores for each of a document's segments and assigns the aggregated score to the document.
        :param documents: A list of Text objects, each of which corresponds to an indexed document.
        :param segments_group: A SegmentGroup containing all the documents' segments and the end index of each document.
        :param method: The aggregation function to use (default is max).
        :return: The updated list of documents, including scores.
        """
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
