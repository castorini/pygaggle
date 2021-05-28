import numpy as np
from collections import defaultdict

from .base import Answer
from .utils import normalize_answer


class SpanSelection:
    def reset(self):
        pass

    def score(self, span, text):
        pass

    def add_answers(self, spans_by_text, texts):
        pass

    def top_answers(self, num_spans):
        pass

    def __str__(self):
        pass


class DprSelection(SpanSelection):
    def reset(self):
        self.answers = []

    def score(self, span, text):
        return float(span.relevance_score), float(span.span_score)

    def add_answers(self, spans_by_text, texts):
        for spans, text in zip(spans_by_text, texts):
            for span in spans:
                self.answers.append(Answer(text=span.text,
                                           context=text,
                                           score=self.score(span, text)))

    def top_answers(self, num_spans):
        return sorted(self.answers, reverse=True, key=lambda answer: answer.score)[: num_spans]

    def __str__(self):
        return 'DPR'


class DprFusionSelection(DprSelection):
    def __init__(self, beta, gamma):
        self.beta = float(beta)
        self.gamma = float(gamma)

    def score(self, span, text):
        return float(span.relevance_score) * self.beta + float(text.score) * self.gamma, float(span.span_score)

    def __str__(self):
        return f'DPR Fusion, beta={self.beta}, gamma={self.gamma}'


class GarSelection(SpanSelection):
    def reset(self):
        self.answers = defaultdict(int)

    def score(self, span, text):
        return float(span.relevance_score)

    def add_answers(self, spans_by_text, texts):
        eD = np.exp(np.array([self.score(spans[0], text) for spans, text in zip(spans_by_text, texts)]))

        for i, spans in enumerate(spans_by_text):
            topn_spans = spans[:5]
            eSi = np.exp(np.array([float(span.span_score) for span in topn_spans]))
            softmaxSi = list(eSi / np.sum(eSi))

            for j, span in enumerate(topn_spans):
                self.answers[normalize_answer(span.text)] += eD[i] * softmaxSi[j]

    def top_answers(self, num_spans):
        answers = sorted(list(self.answers.items()), reverse=True, key=lambda answer: answer[1])[: num_spans]
        return list(map(lambda answer: Answer(text=answer[0], score=answer[1]), answers))

    def __str__(self):
        return 'GAR'


class GarFusionSelection(GarSelection):
    def __init__(self, beta, gamma):
        self.beta = float(beta)
        self.gamma = float(gamma)

    def score(self, span, text):
        return float(span.relevance_score) * self.beta + float(text.score) * self.gamma

    def __str__(self):
        return f'GAR Fusion, beta={self.beta}, gamma={self.gamma}'
