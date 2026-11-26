from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from vtrain.evaluation.deep import (
    bootstrap_ci,
    calibration_report,
    group_metrics,
    quantile_bins,
    slice_gap,
)


def test_bootstrap_ci_range() -> None:
    y = np.array([0, 1] * 50)
    scores = np.array([0.1, 0.9] * 50)
    ci = bootstrap_ci(y, scores, average_precision_score, n_boot=50, seed=0)
    assert ci["low"] <= ci["mean"] <= ci["high"]
    assert ci["n_boot"] > 0


def test_quantile_bins() -> None:
    reference = np.arange(100.0)
    values = np.array([0.0, 50.0, 99.0])
    bins = quantile_bins(reference, values, n_bins=3)
    assert len(bins) == 3
    assert bins[0] < bins[-1]


def test_group_metrics_and_gap() -> None:
    groups = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([0, 1, 0, 0, 1, 0])
    scores = np.array([0.1, 0.9, 0.2, 0.2, 0.8, 0.1])
    gm = group_metrics(groups, y, scores, min_size=2)
    assert set(gm) == {"0", "1"}
    assert "roc_auc" in gm["0"]
    assert slice_gap(gm, "roc_auc") >= 0.0


def test_slice_gap_empty() -> None:
    assert slice_gap({}) == 0.0


def test_calibration_report() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    report = calibration_report(y, proba)
    assert 0.0 <= report["ece"] <= 1.0
    assert report["brier"] >= 0.0
    assert len(report["bins"]) >= 1


def test_calibration_perfect() -> None:
    y = np.array([0, 1])
    proba = np.array([0.0, 1.0])
    report = calibration_report(y, proba)
    assert report["brier"] == pytest.approx(0.0)
