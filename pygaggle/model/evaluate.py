from collections import OrderedDict
from typing import List
import abc

from sklearn.metrics import recall_score
from tqdm import tqdm
import numpy as np

from pygaggle.data import RelevanceExample
from pygaggle.rerank import Reranker


__all__ = ['RerankerEvaluator', 'metric_names']
METRIC_MAP = OrderedDict()


class MetricAccumulator:
    name: str = None

    def accumulate(self, scores: List[float], gold: List[RelevanceExample]):
        return

    @abc.abstractmethod
    def value(self):
        return


class MeanAccumulator(MetricAccumulator):
    def __init__(self):
        self.scores = []

    @property
    def value(self):
        return np.mean(self.scores)


def register_metric(name):
    def wrap_fn(metric_cls):
        METRIC_MAP[name] = metric_cls
        metric_cls.name = name
        return metric_cls
    return wrap_fn


def metric_names():
    return list(METRIC_MAP.keys())


def truncated_rels(scores: List[float], top_k: int) -> np.ndarray:
    rel_idxs = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:top_k]
    rel_idxs = [x[0] for x in rel_idxs]
    score_rels = np.zeros(len(scores), dtype=int)
    score_rels[rel_idxs] = 1
    return score_rels


@register_metric('recall')
class RecallAccumulator(MeanAccumulator):
    top_k = None

    def accumulate(self, scores: List[float], gold: RelevanceExample):
        score_rels = truncated_rels(scores, self.top_k)
        gold_rels = np.array(gold.labels, dtype=int)
        score = recall_score(gold_rels, score_rels, zero_division=1)
        self.scores.append(score)


@register_metric('precision')
class PrecisionAccumulator(MeanAccumulator):
    top_k = None

    def accumulate(self, scores: List[float], gold: RelevanceExample):
        score_rels = truncated_rels(scores, self.top_k)
        gold_rels = np.array(gold.labels, dtype=int)
        self.scores.append((score_rels & gold_rels).sum() / score_rels.sum())


@register_metric('recall@1')
class RecallAt1Metric(RecallAccumulator):
    top_k = 1


@register_metric('precision@1')
class PrecisionAt1Metric(PrecisionAccumulator):
    top_k = 1


class RerankerEvaluator:
    def __init__(self,
                 reranker: Reranker,
                 metric_names: List[str],
                 use_tqdm: bool = True):
        self.reranker = reranker
        self.metrics = [METRIC_MAP[name] for name in metric_names]
        self.use_tqdm = use_tqdm

    def evaluate(self, examples: List[RelevanceExample]) -> List[MetricAccumulator]:
        metrics = [cls() for cls in self.metrics]
        for example in tqdm(examples, disable=not self.use_tqdm):
            scores = [x.score for x in self.reranker.rerank(example.query, example.documents)]
            for metric in metrics:
                metric.accumulate(scores, example)
        return metrics
