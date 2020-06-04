import spacy
from pygaggle.rerank.base import Text
from typing import List

def segment(documents: List[Text]):
    stride = 5
    max_length = 10
    segmented_doc, doc_len, end = [], [0], 0
    
    nlp = spacy.blank("en")
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    for document in documents:
        content = document.text
        doc = nlp(content)
        sentences = [sent.string.strip() for sent in doc.sents]
        for i in range(0, len(sentences), stride):
            segment = ' '.join(sentences[i:i + max_length])
            segmented_doc.append(Text(segment,dict(docid=document.raw["docid"])))
            if i + max_length >= len(sentences):
                end += i/stride + 1
                doc_len.append(int(end))
                break

    return segmented_doc, doc_len

def aggregate(scores: List[int], doc_len: List[int]): 

    aggregated_scores = [max(scores[doc_len[i]:doc_len[i+1]]) for i in range(len(doc_len)-1)]
    
    return aggregated_scores
