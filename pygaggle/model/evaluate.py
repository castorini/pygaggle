from collections import OrderedDict
from typing import List, Optional
import abc

from sklearn.metrics import recall_score
from tqdm import tqdm
import numpy as np

from pygaggle.data.kaggle import RelevanceExample
from pygaggle.rerank.base import Reranker
from pygaggle.model.writer import Writer

from pygaggle.data.segmentation import SegmentProcessor

__all__ = ['RerankerEvaluator', 'DuoRerankerEvaluator', 'metric_names']
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
        rel_idxs = sorted(list(enumerate(scores)),
                          key=lambda x: x[1], reverse=True)[self.top_k:]
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
        score = recall_score(gold_rels, score_rels, zero_division=0)
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


@register_metric('recall@50')
class RecallAt50Metric(TopkMixin, RecallAccumulator):
    top_k = 50


@register_metric('recall@1000')
class RecallAt1000Metric(TopkMixin, RecallAccumulator):
    top_k = 1000


@register_metric('mrr')
class MrrMetric(MeanAccumulator):
    def accumulate(self, scores: List[float], gold: RelevanceExample):
        scores = sorted(list(enumerate(scores)),
                        key=lambda x: x[1], reverse=True)
        rr = next((1 / (rank_idx + 1) for rank_idx, (idx, _) in
                   enumerate(scores) if gold.labels[idx]), 0)
        self.scores.append(rr)


@register_metric('mrr@10')
class MrrAt10Metric(MeanAccumulator):
    def accumulate(self, scores: List[float], gold: RelevanceExample):
        scores = sorted(list(enumerate(scores)), key=lambda x: x[1],
                        reverse=True)
        rr = next((1 / (rank_idx + 1) for rank_idx, (idx, _) in
                   enumerate(scores) if (gold.labels[idx] and rank_idx < 10)),
                  0)
        self.scores.append(rr)


class ThresholdedRecallMetric(DynamicThresholdingMixin, RecallAccumulator):
    threshold = 0.5


class ThresholdedPrecisionMetric(DynamicThresholdingMixin,
                                 PrecisionAccumulator):
    threshold = 0.5


class RerankerEvaluator:
    def __init__(self,
                 reranker: Reranker,
                 metric_names: List[str],
                 use_tqdm: bool = True,
                 writer: Optional[Writer] = None):
        self.reranker = reranker
        self.metrics = [METRIC_MAP[name] for name in metric_names]
        self.use_tqdm = use_tqdm
        self.writer = writer

    def evaluate(self,
                 examples: List[RelevanceExample]) -> List[MetricAccumulator]:
        metrics = [cls() for cls in self.metrics]
        for example in tqdm(examples, disable=not self.use_tqdm):
            scores = [x.score for x in self.reranker.rerank(example.query,
                                                            example.documents)]
            if self.writer is not None:
                self.writer.write(scores, example)
            for metric in metrics:
                metric.accumulate(scores, example)
        return metrics

    def evaluate_by_segments(self,
                             examples: List[RelevanceExample],
                             seg_size: int,
                             stride: int,
                             aggregate_method: str) -> List[MetricAccumulator]:
        metrics = [cls() for cls in self.metrics]
        segment_processor = SegmentProcessor()
        for example in tqdm(examples, disable=not self.use_tqdm):
            segment_group = segment_processor.segment(example.documents, seg_size, stride)
            segment_group.segments = self.reranker.rerank(example.query, segment_group.segments)
            doc_scores = [x.score for x in segment_processor.aggregate(example.documents,
                                                                       segment_group,
                                                                       aggregate_method)]
            if self.writer is not None:
                self.writer.write(doc_scores, example)
            for metric in metrics:
                metric.accumulate(doc_scores, example)
        return metrics


class DuoRerankerEvaluator:
    def __init__(self,
                 mono_reranker: Reranker,
                 duo_reranker: Reranker,
                 metric_names: List[str],
                 mono_hits: int = 50,
                 use_tqdm: bool = True,
                 writer: Optional[Writer] = None):
        self.mono_reranker = mono_reranker
        self.duo_reranker = duo_reranker
        self.mono_hits = mono_hits
        self.metrics = [METRIC_MAP[name] for name in metric_names]
        self.use_tqdm = use_tqdm
        self.writer = writer

    def evaluate(self,
                 examples: List[RelevanceExample]) -> List[MetricAccumulator]:
        metrics = [cls() for cls in self.metrics]
        mono_texts = []
        scores = []
        for ct, example in tqdm(enumerate(examples), total=len(examples), disable=not self.use_tqdm):
            mono_out = self.mono_reranker.rerank(example.query, example.documents)
            mono_texts.append(sorted(enumerate(mono_out), key=lambda x: x[1].score, reverse=True)[:self.mono_hits])
            scores.append(np.array([x.score for x in mono_out]))
        for ct, texts in tqdm(enumerate(mono_texts), total=len(mono_texts), disable=not self.use_tqdm):
            duo_in = list(map(lambda x: x[1], texts))
            duo_scores = [x.score for x in self.duo_reranker.rerank(examples[ct].query, duo_in)]

            scores[ct][list(map(lambda x: x[0], texts))] = duo_scores
            if self.writer is not None:
                self.writer.write(list(scores[ct]), examples[ct])
            for metric in metrics:
                metric.accumulate(list(scores[ct]), examples[ct])
        return metrics

    def evaluate_by_segments(self,
                             examples: List[RelevanceExample],
                             seg_size: int,
                             stride: int,
                             aggregate_method: str) -> List[MetricAccumulator]:
        metrics = [cls() for cls in self.metrics]
        segment_processor = SegmentProcessor()
        for example in tqdm(examples, disable=not self.use_tqdm):
            segment_group = segment_processor.segment(example.documents, seg_size, stride)
            segment_group.segments = self.reranker.rerank(example.query, segment_group.segments)
            doc_scores = [x.score for x in segment_processor.aggregate(example.documents,
                                                                       segment_group,
                                                                       aggregate_method)]
            if self.writer is not None:
                self.writer.write(doc_scores, example)
            for metric in metrics:
                metric.accumulate(doc_scores, example)
        return metrics