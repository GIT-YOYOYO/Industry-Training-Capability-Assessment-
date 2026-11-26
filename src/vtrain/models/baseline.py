"""朴素基线：Dummy（先验概率）与规则打分（基线 5 维等权平均的负值）。

基线是“模型分数值不值”的参照。PR-AUC 只看排序，所以规则基线用等权平均即可，
不追求校准——它只用于对比，不部署。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from vtrain.config import AppConfig, FeatureSpec
from vtrain.features.build import FeatureSelector
from vtrain.models.registry import register


@register("dummy")
def _make_dummy(cfg: AppConfig, spec: FeatureSpec) -> Pipeline:
    return Pipeline(
        [
            ("select", FeatureSelector(spec.feature_columns)),
            ("clf", DummyClassifier(strategy="prior", random_state=cfg.seed)),
        ]
    )


class RuleBaselineClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """风险分数 = -mean(基线维度)。predict_proba 用 logistic 单调映射，仅供排序/决策。"""

    def fit(self, X: object, y: object = None) -> RuleBaselineClassifier:
        scores = self._score(X)
        self.center_ = float(np.median(scores))
        self.classes_ = np.array([0, 1])
        return self

    def _score(self, X: object) -> NDArray[np.float64]:
        arr = np.asarray(X, dtype=np.float64)
        return -arr.mean(axis=1)

    def predict_proba(self, X: object) -> NDArray[np.float64]:
        score = self._score(X) - self.center_
        proba = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack([1.0 - proba, proba]).astype(np.float64)

    def predict(self, X: object) -> NDArray[np.int64]:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int64)


@register("rule_baseline")
def _make_rule(cfg: AppConfig, spec: FeatureSpec) -> Pipeline:
    return Pipeline(
        [
            ("select", FeatureSelector(spec.feature_columns)),
            ("clf", RuleBaselineClassifier()),
        ]
    )
