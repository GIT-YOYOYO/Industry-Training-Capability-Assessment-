from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from vtrain.explain.shap_explain import explain, global_importance, raw_score
from vtrain.models import build_model


@pytest.fixture()
def fitted(toy_project: Any) -> dict[str, tuple[BaseEstimator, pd.DataFrame]]:
    _, cfg, spec = toy_project
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4", "x5"])
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    models: dict[str, tuple[BaseEstimator, pd.DataFrame]] = {}
    for name in ("logreg", "lightgbm"):
        est = build_model(name, cfg, spec)
        est.fit(df, y)
        models[name] = (est, df)
    return models


@pytest.mark.parametrize("name", ["logreg", "lightgbm"])
def test_additivity_matches_raw_score(
    fitted: dict[str, tuple[BaseEstimator, pd.DataFrame]], name: str
) -> None:
    est, df = fitted[name]
    sample = df.head(10)
    exps = explain(est, sample, ["x1", "x2", "x3", "x4", "x5"])
    raw = raw_score(est, sample)
    for i, exp in enumerate(exps):
        assert exp.total == pytest.approx(float(raw[i]), abs=1e-6)


def test_global_importance_sorted(fitted: dict[str, tuple[BaseEstimator, pd.DataFrame]]) -> None:
    est, df = fitted["logreg"]
    exps = explain(est, df.head(50), ["x1", "x2", "x3", "x4", "x5"])
    ranking = global_importance(exps)
    values = [value for _, value in ranking]
    assert values == sorted(values, reverse=True)
    assert {name for name, _ in ranking} == {"x1", "x2", "x3", "x4", "x5"}
