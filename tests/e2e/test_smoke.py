"""端到端冒烟：在现造的小项目上跑通 prepare -> train -> predict，并验证可复现。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vtrain.bundle.bundle import load_bundle
from vtrain.config import AppConfig, FeatureSpec
from vtrain.data.prepare import prepare
from vtrain.explain.shap_explain import raw_score
from vtrain.models.train import train
from vtrain.predict import predict_frame


def test_smoke_end_to_end(toy_project: tuple[Path, AppConfig, FeatureSpec]) -> None:
    root, cfg, spec = toy_project
    manifest = prepare(cfg, spec, root)
    assert manifest.group_disjoint
    assert set(manifest.splits) == {"train", "val", "test"}

    result = train(cfg, spec, root)
    assert result.report["acceptance"]["passed"]

    pipeline, meta = load_bundle(result.model_dir, spec.feature_columns)
    df = pd.read_parquet(root / "data/processed/test.parquet").head(3)
    rows = predict_frame(pipeline, meta, df)
    assert len(rows) == 3
    raw = raw_score(pipeline, df)
    for i, row in enumerate(rows):
        assert 0.0 <= row["proba_low_ability"] <= 1.0
        assert abs(row["base_value"] + sum(row["contributions"].values()) - row["logit"]) < 1e-6
        assert row["logit"] == pytest.approx(float(raw[i]), abs=1e-6)


def test_train_is_deterministic(toy_project: tuple[Path, AppConfig, FeatureSpec]) -> None:
    root, cfg, spec = toy_project
    prepare(cfg, spec, root)
    first = train(cfg, spec, root)
    second = train(cfg, spec, root)
    assert first.champion == second.champion
    assert first.report["champion_test"]["pr_auc"] == second.report["champion_test"]["pr_auc"]
