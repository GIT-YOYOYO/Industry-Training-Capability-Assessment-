from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from vtrain.bundle.bundle import BundleError, load_bundle, save_bundle


def _fitted() -> LogisticRegression:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = (X[:, 0] > 0).astype(int)
    return LogisticRegression().fit(X, y)


def _meta() -> dict[str, object]:
    return {
        "model_name": "logreg",
        "feature_columns": ["x1", "x2"],
        "decision_rule": {"strategy": "top_k", "k_fraction": 0.1, "threshold": 0.5},
        "contribution_unit": "log_odds",
        "config_fingerprint": "abc",
        "label_threshold": 0.0,
        "metrics": {},
    }


def test_roundtrip(tmp_path: Path) -> None:
    save_bundle(tmp_path, _fitted(), _meta())
    est, meta = load_bundle(tmp_path, ("x1", "x2"))
    assert meta["model_name"] == "logreg"
    assert hasattr(est, "predict_proba")


def test_feature_mismatch_rejected(tmp_path: Path) -> None:
    save_bundle(tmp_path, _fitted(), _meta())
    with pytest.raises(BundleError):
        load_bundle(tmp_path, ("x1", "x9"))


def test_missing_metadata_rejected(tmp_path: Path) -> None:
    save_bundle(tmp_path, _fitted(), {"model_name": "x"})
    with pytest.raises(BundleError):
        load_bundle(tmp_path)


def test_missing_files_rejected(tmp_path: Path) -> None:
    with pytest.raises(BundleError):
        load_bundle(tmp_path / "nope")
