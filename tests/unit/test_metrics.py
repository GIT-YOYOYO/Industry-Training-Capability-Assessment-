from __future__ import annotations

import numpy as np
import pytest

from vtrain.evaluation.metrics import (
    classification_metrics,
    top_k_count,
    top_k_mask,
    top_k_recall,
)


def test_top_k_count() -> None:
    assert top_k_count(100, 0.1) == 10
    assert top_k_count(10, 0.1) == 1
    assert top_k_count(5667, 0.1) == 567


def test_top_k_mask() -> None:
    mask = top_k_mask([0.1, 0.9, 0.5], 2)
    assert mask.tolist() == [False, True, True]


def test_top_k_recall_perfect() -> None:
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.2])
    assert top_k_recall(y, scores, 0.5) == 1.0


def test_metrics_keys_and_perfect_pr_auc() -> None:
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = classification_metrics(y, scores, 0.5)
    assert {"pr_auc", "roc_auc", "brier", "top_k_recall", "positive_rate"} <= set(metrics)
    assert metrics["pr_auc"] == pytest.approx(1.0)
