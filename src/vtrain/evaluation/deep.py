"""深入评估（X2）：自助法置信区间、分组切片、概率校准、错误分析。

这些指标回答“这个数字为什么可信、在哪些人群上不稳”，而不是只报一个分数。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
Metric = Callable[[IntArray, FloatArray], float]


class CalibrationBin(TypedDict):
    bin: float
    n: float
    confidence: float
    accuracy: float


class CalibrationReport(TypedDict):
    brier: float
    ece: float
    bins: list[CalibrationBin]


def bootstrap_ci(
    y_true: Sequence[int] | IntArray,
    scores: Sequence[float] | FloatArray,
    metric: Metric,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, float]:
    """对指标做自助法重采样，返回点估计的均值与 (1-alpha) 区间。"""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        stats.append(float(metric(yb, s[idx])))
    arr = np.asarray(stats, dtype=np.float64)
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(arr.mean()),
        "low": float(lo),
        "high": float(hi),
        "n_boot": float(len(arr)),
    }


def quantile_bins(
    reference: Sequence[float] | FloatArray,
    values: Sequence[float] | FloatArray,
    n_bins: int,
) -> IntArray:
    """用 reference（训练集）的分位点切箱，再作用到 values（避免用评估集定边界）。"""
    edges = np.quantile(
        np.asarray(reference, dtype=np.float64), np.linspace(0, 1, n_bins + 1)[1:-1]
    )
    return np.digitize(np.asarray(values, dtype=np.float64), edges).astype(np.int64)


def group_metrics(
    groups: Sequence[int] | IntArray,
    y_true: Sequence[int] | IntArray,
    scores: Sequence[float] | FloatArray,
    min_size: int,
) -> dict[str, dict[str, float]]:
    """每个分组的样本量、正例率、PR-AUC、ROC-AUC、lift(PR/正例率)；样本量不足的组跳过。"""
    g = np.asarray(groups, dtype=np.int64)
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    out: dict[str, dict[str, float]] = {}
    for value in np.unique(g):
        mask = g == value
        n = int(mask.sum())
        if n < min_size:
            continue
        yy, ss = y[mask], s[mask]
        single_class = yy.min() == yy.max()
        pr_auc = float("nan") if single_class else float(average_precision_score(yy, ss))
        roc_auc = float("nan") if single_class else float(roc_auc_score(yy, ss))
        prevalence = float(yy.mean())
        out[str(int(value))] = {
            "n": float(n),
            "positive_rate": prevalence,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "lift": pr_auc / prevalence if prevalence > 0 else float("nan"),
        }
    return out


def slice_gap(gm: dict[str, dict[str, float]], metric: str = "roc_auc") -> float:
    vals = [v[metric] for v in gm.values() if np.isfinite(v.get(metric, float("nan")))]
    return float(max(vals) - min(vals)) if len(vals) > 1 else 0.0


def calibration_report(
    y_true: Sequence[int] | IntArray,
    proba: Sequence[float] | FloatArray,
    n_bins: int = 10,
) -> CalibrationReport:
    """Brier + ECE + 每个概率箱的置信度/实际正例率。"""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(proba, dtype=np.float64)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows: list[CalibrationBin] = []
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        count = int(mask.sum())
        ece += abs(acc - conf) * count / len(y)
        rows.append({"bin": float(b), "n": float(count), "confidence": conf, "accuracy": acc})
    return {
        "brier": float(brier_score_loss(y, np.clip(p, 0.0, 1.0))),
        "ece": float(ece),
        "bins": rows,
    }


def error_examples(
    df: pd.DataFrame,
    y_true: Sequence[int] | IntArray,
    proba: Sequence[float] | FloatArray,
    decision: Sequence[int] | IntArray,
    feature_columns: Sequence[str],
    top_n: int = 5,
) -> dict[str, object]:
    """挑出“最自信的漏报”和“最可疑的误报”，只保留特征值，不含标识符。"""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(proba, dtype=np.float64)
    d = np.asarray(decision, dtype=np.int64)

    def pack(indices: IntArray) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for i in indices.tolist():
            rows.append(
                {
                    "proba": float(p[i]),
                    "features": {col: float(df.iloc[i][col]) for col in feature_columns},
                }
            )
        return rows

    fn_idx = np.where((y == 1) & (d == 0))[0]
    fn_idx = fn_idx[np.argsort(p[fn_idx])][:top_n]
    fp_idx = np.where((y == 0) & (d == 1))[0]
    fp_idx = fp_idx[np.argsort(-p[fp_idx])][:top_n]
    return {
        "n_false_negative": int(((y == 1) & (d == 0)).sum()),
        "n_false_positive": int(((y == 0) & (d == 1)).sum()),
        "false_negatives": pack(fn_idx),
        "false_positives": pack(fp_idx),
    }
