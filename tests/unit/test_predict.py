"""推理入口的分支与错误路径：读入格式、predict_file 往返、缺列报错。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vtrain.bundle.bundle import save_bundle
from vtrain.features.build import FeatureContractError
from vtrain.models import build_model
from vtrain.predict import predict_file, read_input


@pytest.fixture()
def trained_bundle(toy_project: Any) -> tuple[Path, pd.DataFrame]:
    root, cfg, spec = toy_project
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4", "x5"])
    y = (X[:, 0] > 0).astype(int)
    est = build_model("logreg", cfg, spec)
    est.fit(df, y)
    meta = {
        "model_name": "logreg",
        "feature_columns": list(spec.feature_columns),
        "decision_rule": {"strategy": "top_k", "k_fraction": 0.1, "threshold": 0.5},
        "contribution_unit": "log_odds",
        "config_fingerprint": "x",
        "label_threshold": 0.0,
    }
    model_dir = root / "models" / "champ"
    save_bundle(model_dir, est, meta)
    return model_dir, df


def test_read_input_supports_csv_parquet_xlsx(
    tmp_path: Path, trained_bundle: tuple[Path, pd.DataFrame]
) -> None:
    _, df = trained_bundle
    csv = tmp_path / "x.csv"
    parquet = tmp_path / "x.parquet"
    xlsx = tmp_path / "x.xlsx"
    df.to_csv(csv, index=False)
    df.to_parquet(parquet, index=False)
    df.to_excel(xlsx, index=False)
    for path in (csv, parquet, xlsx):
        assert len(read_input(path)) == len(df)


def test_read_input_rejects_unsupported(tmp_path: Path) -> None:
    bad = tmp_path / "x.txt"
    bad.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError):
        read_input(bad)


def test_predict_file_roundtrip(tmp_path: Path, trained_bundle: tuple[Path, pd.DataFrame]) -> None:
    model_dir, df = trained_bundle
    csv = tmp_path / "in.csv"
    df.head(3).to_csv(csv, index=False)
    rows = predict_file(model_dir, csv)
    assert len(rows) == 3
    assert all(0.0 <= row["proba_low_ability"] <= 1.0 for row in rows)


def test_predict_file_missing_column_raises(
    tmp_path: Path, trained_bundle: tuple[Path, pd.DataFrame]
) -> None:
    model_dir, df = trained_bundle
    bad = tmp_path / "bad.csv"
    df.head(3).drop(columns=["x1"]).to_csv(bad, index=False)
    with pytest.raises(FeatureContractError):
        predict_file(model_dir, bad)
