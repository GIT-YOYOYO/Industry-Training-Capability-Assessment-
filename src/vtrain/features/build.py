"""特征构建：训练与推理共用同一套列选择与校验。

FeatureSelector 是 sklearn 兼容的转换器，放在每个 Pipeline 的第一环：
- 只保留 configs/features.yaml 契约内的列；
- 缺列立即报错，绝不静默填充；
- 推理时同一对象保证口径与训练一致。
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureContractError(ValueError):
    """输入缺少契约要求的特征列，或特征含缺失。"""


class FeatureSelector(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)

    def fit(self, X: pd.DataFrame, y: object = None) -> FeatureSelector:
        self._check(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check(X)
        return X.loc[:, list(self.columns)].copy()

    def _check(self, X: pd.DataFrame) -> None:
        missing = [col for col in self.columns if col not in X.columns]
        if missing:
            raise FeatureContractError(f"输入缺少特征列: {missing}")
        na_total = int(X.loc[:, list(self.columns)].isna().sum().sum())
        if na_total:
            raise FeatureContractError(f"特征列存在缺失值 {na_total} 个，拒绝静默填充")


def select_features(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """非 Pipeline 场景下的直接列选择（评估切片等）。"""
    return FeatureSelector(columns).transform(df)
