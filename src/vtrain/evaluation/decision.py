"""决策规则：名额制（Top-K）。

阈值只在验证集上定，随模型一起保存；应用到新数据时用固定阈值，
不再看新数据的分数分布（避免把测试集的分布信息带进决策）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vtrain.evaluation.metrics import top_k_count


@dataclass(frozen=True)
class DecisionRule:
    strategy: str
    k_fraction: float
    threshold: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "strategy": self.strategy,
            "k_fraction": self.k_fraction,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DecisionRule:
        return cls(
            strategy=str(data["strategy"]),
            k_fraction=float(str(data["k_fraction"])),
            threshold=float(str(data["threshold"])),
        )


def fit_top_k_rule(
    val_scores: Sequence[float] | NDArray[np.float64], k_fraction: float
) -> DecisionRule:
    """阈值 = 验证集分数第 k 高的值；分数 >= 阈值即进入强化班。"""
    arr = np.asarray(val_scores, dtype=np.float64)
    k = top_k_count(len(arr), k_fraction)
    ordered = np.sort(arr)[::-1]
    return DecisionRule(strategy="top_k", k_fraction=k_fraction, threshold=float(ordered[k - 1]))


def apply_rule(
    scores: Sequence[float] | NDArray[np.float64], rule: DecisionRule
) -> NDArray[np.int64]:
    arr = np.asarray(scores, dtype=np.float64)
    return (arr >= rule.threshold).astype(np.int64)


def selected_fraction(decisions: Sequence[int] | NDArray[np.int64]) -> float:
    arr = np.asarray(decisions, dtype=np.int64)
    return float(arr.mean()) if len(arr) else 0.0


def decision_outcome(
    y_true: Sequence[int] | NDArray[np.int64],
    decisions: Sequence[int] | NDArray[np.int64],
) -> dict[str, float]:
    """固定阈值决策后的效果：选中比例、召回、精确率、人数。"""
    y = np.asarray(y_true, dtype=np.int64)
    sel = np.asarray(decisions, dtype=np.int64) == 1
    n_pos = int((y == 1).sum())
    n_selected = int(sel.sum())
    true_positive = int((sel & (y == 1)).sum())
    return {
        "selected_fraction": float(sel.mean()) if len(y) else 0.0,
        "recall": float(true_positive / n_pos) if n_pos else float("nan"),
        "precision": float(true_positive / n_selected) if n_selected else 0.0,
        "n_selected": float(n_selected),
        "n_true_positive": float(true_positive),
    }
