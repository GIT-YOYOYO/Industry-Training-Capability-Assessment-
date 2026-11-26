"""LightGBM 候选：树模型不需要标准化，Pipeline 只有列选择 + 分类器。"""

from __future__ import annotations

import lightgbm as lgb
from sklearn.pipeline import Pipeline

from vtrain.config import AppConfig, FeatureSpec
from vtrain.features.build import FeatureSelector
from vtrain.models.registry import register


@register("lightgbm")
def make_lightgbm(cfg: AppConfig, spec: FeatureSpec) -> Pipeline:
    params = cfg.models.lightgbm
    clf = lgb.LGBMClassifier(
        n_estimators=params.n_estimators,
        learning_rate=params.learning_rate,
        num_leaves=params.num_leaves,
        min_child_samples=params.min_child_samples,
        subsample=params.subsample,
        subsample_freq=params.subsample_freq,
        colsample_bytree=params.colsample_bytree,
        reg_lambda=params.reg_lambda,
        n_jobs=params.n_jobs,
        random_state=cfg.seed,
        verbose=-1,
    )
    return Pipeline(
        [
            ("select", FeatureSelector(spec.feature_columns)),
            ("clf", clf),
        ]
    )
