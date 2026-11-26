"""逻辑回归候选：标准化 + L2 正则，class_weight=balanced 处理 25% 正例的不平衡。"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vtrain.config import AppConfig, FeatureSpec
from vtrain.features.build import FeatureSelector
from vtrain.models.registry import register


@register("logreg")
def make_logreg(cfg: AppConfig, spec: FeatureSpec) -> Pipeline:
    params = cfg.models.logreg
    return Pipeline(
        [
            ("select", FeatureSelector(spec.feature_columns)),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=params.C,
                    max_iter=params.max_iter,
                    class_weight=params.class_weight,
                    solver="lbfgs",
                    random_state=cfg.seed,
                ),
            ),
        ]
    )
