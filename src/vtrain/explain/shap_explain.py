"""可加性解释。

两条实现路线（课程做法，不依赖 shap 库）：
- 逻辑回归：贡献_j = coef_j * 标准化后取值，基准值 = 截距。因标准化后训练均值=0，
  基准值对应“平均学员”，贡献单位是对数几率。
- LightGBM：booster_.predict(pred_contrib=True) 的 TreeSHAP，最后一列是基准值。

两条路线都满足：基准值 + 全部贡献 = 模型原始输出（对数几率），有测试守着。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

CONTRIBUTION_UNIT = "log_odds"


@dataclass(frozen=True)
class Explanation:
    base_value: float
    contributions: dict[str, float]

    @property
    def total(self) -> float:
        """基准值 + 全部贡献 = 模型原始输出（对数几率）。"""
        return self.base_value + sum(self.contributions.values())

    def ranked(self, top_n: int | None = None) -> list[tuple[str, float]]:
        """按贡献绝对值从大到小排序（|贡献| 越大越关键）。"""
        items = sorted(self.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return items if top_n is None else items[:top_n]


def _named(est: BaseEstimator) -> dict[str, Any]:
    return cast("dict[str, Any]", dict(est.named_steps))


def _select(est: BaseEstimator, X: pd.DataFrame) -> pd.DataFrame:
    selector = _named(est)["select"]
    return cast("pd.DataFrame", selector.transform(X))


def _clf(est: BaseEstimator) -> Any:
    return _named(est)["clf"]


def raw_score(est: BaseEstimator, X: pd.DataFrame) -> NDArray[np.float64]:
    """模型原始输出（对数几率空间）。线性模型用 coef·x+intercept，树模型用 raw_score。"""
    clf = _clf(est)
    X_sel = _select(est, X)
    if hasattr(clf, "booster_"):
        raw = np.asarray(clf.booster_.predict(X_sel, raw_score=True), dtype=np.float64)
        return cast("NDArray[np.float64]", raw.reshape(-1))
    steps = _named(est)
    if "scale" in steps:
        X_sel = cast("pd.DataFrame", steps["scale"].transform(X_sel))
    coef = np.asarray(clf.coef_, dtype=np.float64)[0]
    intercept = float(np.asarray(clf.intercept_, dtype=np.float64)[0])
    result = np.asarray(X_sel, dtype=np.float64) @ coef + intercept
    return cast("NDArray[np.float64]", result)


def explain(
    est: BaseEstimator,
    X: pd.DataFrame,
    feature_columns: Sequence[str],
) -> list[Explanation]:
    """对每一行给出基准值与各特征贡献（单位：对数几率）。"""
    clf = _clf(est)
    X_sel = _select(est, X)
    names = list(feature_columns)
    if hasattr(clf, "booster_"):
        contrib = np.asarray(clf.booster_.predict(X_sel, pred_contrib=True), dtype=np.float64)
        bases = contrib[:, -1]
        feat_contrib = contrib[:, :-1]
    else:
        steps = _named(est)
        if "scale" in steps:
            X_trans = np.asarray(steps["scale"].transform(X_sel), dtype=np.float64)
        else:
            X_trans = np.asarray(X_sel, dtype=np.float64)
        coef = np.asarray(clf.coef_, dtype=np.float64)[0]
        base = float(np.asarray(clf.intercept_, dtype=np.float64)[0])
        bases = np.full(len(X_trans), base, dtype=np.float64)
        feat_contrib = X_trans * coef
    return [
        Explanation(
            base_value=float(bases[i]),
            contributions={name: float(feat_contrib[i, j]) for j, name in enumerate(names)},
        )
        for i in range(len(X_sel))
    ]


def global_importance(explanations: Sequence[Explanation]) -> list[tuple[str, float]]:
    """全局解释：按平均 |贡献| 排序。"""
    if not explanations:
        return []
    names = list(explanations[0].contributions)
    means = {
        name: float(np.mean([abs(exp.contributions[name]) for exp in explanations]))
        for name in names
    }
    return sorted(means.items(), key=lambda kv: kv[1], reverse=True)
