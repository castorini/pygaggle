from collections import OrderedDict
from typing import List
import abc

from sklearn.metrics import recall_score
from tqdm import tqdm
import numpy as np

from pygaggle.data.kaggle import RelevanceExample
from pygaggle.rerank.base import Reranker


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


class TruncatingMixin:
    def truncated_rels(self, scores: List[float]) -> np.ndarray:
        return np.array(scores)


def register_metric(name):
    def wrap_fn(metric_cls):
        METRIC_MAP[name] = metric_cls
        metric_cls.name = name
        return metric_cls
    return wrap_fn


def metric_names():
    return list(METRIC_MAP.keys())


class TopkMixin(TruncatingMixin):
    top_k: int = None

    def truncated_rels(self, scores: List[float]) -> np.ndarray:
        rel_idxs = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[self.top_k:]
        scores = np.array(scores)
        scores[[x[0] for x in rel_idxs]] = 0
        return scores


class DynamicThresholdingMixin(TruncatingMixin):
    threshold: float = 0.5

    def truncated_rels(self, scores: List[float]) -> np.ndarray:
        scores = np.array(scores)
        scores[scores < self.threshold * np.max(scores)] = 0
        return scores


class RecallAccumulator(TruncatingMixin, MeanAccumulator):
    def accumulate(self, scores: List[float], gold: RelevanceExample):
        score_rels = self.truncated_rels(scores)
        score_rels[score_rels != 0] = 1
        gold_rels = np.array(gold.labels, dtype=int)
        score = recall_score(gold_rels, score_rels, zero_division=1)
        self.scores.append(score)


class PrecisionAccumulator(TruncatingMixin, MeanAccumulator):
    def accumulate(self, scores: List[float], gold: RelevanceExample):
        score_rels = self.truncated_rels(scores)
        score_rels[score_rels != 0] = 1
        score_rels = score_rels.astype(int)
        gold_rels = np.array(gold.labels, dtype=int)
        sum_score = score_rels.sum()
        if sum_score > 0:
            self.scores.append((score_rels & gold_rels).sum() / sum_score)


@register_metric('precision@1')
class PrecisionAt1Metric(TopkMixin, PrecisionAccumulator):
    top_k = 1


@register_metric('recall@3')
class RecallAt3Metric(TopkMixin, RecallAccumulator):
    top_k = 3


@register_metric('mrr')
class MrrMetric(MeanAccumulator):
    def accumulate(self, scores: List[float], gold: RelevanceExample):
        scores = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
        rr = next((1 / (rank_idx + 1) for rank_idx, (idx, _) in enumerate(scores) if gold.labels[idx]), 0)
        self.scores.append(rr)


class ThresholdedRecallMetric(DynamicThresholdingMixin, RecallAccumulator):
    threshold = 0.5


class ThresholdedPrecisionMetric(DynamicThresholdingMixin, PrecisionAccumulator):
    threshold = 0.5


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
