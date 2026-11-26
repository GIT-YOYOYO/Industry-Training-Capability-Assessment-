"""二分类与名额制（Top-K）指标。

主指标 PR-AUC（average precision）：正例约 25%，PR-AUC 比 ROC-AUC 更敏感于头部排序，
与“名额有限、优先抓最该辅导的人”这一决策一致。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ScoreArray = NDArray[np.float64]
LabelArray = NDArray[np.int64]


def top_k_count(n: int, k_fraction: float) -> int:
    k = round(k_fraction * n)
    return max(1, min(k, n))


def top_k_mask(scores: Sequence[float] | ScoreArray, k: int) -> NDArray[np.bool_]:
    arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-arr, kind="stable")
    mask = np.zeros(len(arr), dtype=bool)
    mask[order[: min(k, len(arr))]] = True
    return mask


def top_k_recall(y_true: LabelArray, scores: ScoreArray, k_fraction: float) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    pos = y == 1
    n_pos = int(pos.sum())
    if n_pos == 0:
        return float("nan")
    k = top_k_count(len(y), k_fraction)
    chosen = top_k_mask(scores, k)
    return float((chosen & pos).sum() / n_pos)


def top_k_precision(y_true: LabelArray, scores: ScoreArray, k_fraction: float) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    k = top_k_count(len(y), k_fraction)
    chosen = top_k_mask(scores, k)
    return float((chosen & (y == 1)).sum() / k)


def classification_metrics(
    y_true: Sequence[int] | LabelArray,
    proba: Sequence[float] | ScoreArray,
    k_fraction: float,
) -> dict[str, float]:
    """完整的指标包：主指标 + 辅助 + 名额制 + 正例率。proba 应是 [0,1] 概率。"""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(proba, dtype=np.float64)
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "brier": float(brier_score_loss(y, np.clip(s, 0.0, 1.0))),
        "top_k_recall": top_k_recall(y, s, k_fraction),
        "top_k_precision": top_k_precision(y, s, k_fraction),
        "positive_rate": float(y.mean()),
    }
