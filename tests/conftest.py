"""测试夹具：全部用现造的小数据，不读真实数据。

toy_project 在 tmp 目录里造一份最小项目：24 所学校的小 xlsx + 改过路径的配置，
让 prepare/train 能在几秒内跑通，同时保持与真实配置相同的列契约和流程。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from vtrain.config import AppConfig, FeatureSpec, load_config, load_feature_spec
from vtrain.data.schema import RAW_COLUMNS

REAL_ROOT = Path(__file__).resolve().parents[1]


def _toy_raw() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n_schools = 24
    per_school = 25
    n = n_schools * per_school
    X = rng.normal(size=(n, 5))
    weights = np.array([0.2, 0.25, 0.27, 0.26, 0.24])
    school_effect = np.repeat(rng.normal(scale=0.4, size=n_schools), per_school)
    signal = X @ weights + school_effect
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4", "x5"])
    df["schoolid"] = np.repeat(np.arange(n_schools), per_school)
    df["schooltype"] = np.repeat(np.arange(n_schools) % 3, per_school)
    df["stuid"] = np.arange(100000, 100000 + n)
    df["x"] = X @ weights
    for i in range(1, 6):
        df[f"y{i}"] = signal + rng.normal(scale=0.1, size=n)
    df["y"] = signal
    return df[list(RAW_COLUMNS)]


@pytest.fixture()
def toy_raw() -> pd.DataFrame:
    return _toy_raw()


@pytest.fixture()
def toy_project(tmp_path: Path, toy_raw: pd.DataFrame) -> tuple[Path, AppConfig, FeatureSpec]:
    root = tmp_path
    (root / "configs").mkdir()
    (root / "data/raw").mkdir(parents=True)
    raw_path = root / "data/raw/toy.xlsx"
    toy_raw.to_excel(raw_path, index=False)
    sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    cfg_dict = yaml.safe_load((REAL_ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    cfg_dict["data"]["raw_path"] = "data/raw/toy.xlsx"
    cfg_dict["data"]["raw_sha256"] = sha
    cfg_dict["data"]["test_size"] = 0.25
    cfg_dict["data"]["val_size"] = 0.25
    cfg_dict["evaluation"]["acceptance"]["min_pr_auc_over_baseline"] = -10.0
    cfg_dict["evaluation"]["acceptance"]["max_slice_gap"] = 1.0
    cfg_dict["evaluation"]["bootstrap"]["n"] = 50
    cfg_dict["evaluation"]["slices"]["min_group_size"] = 5
    cfg_dict["models"]["lightgbm"]["n_estimators"] = 40
    cfg_dict["models"]["lightgbm"]["n_jobs"] = 1
    cfg_dict["tracking"]["backend"] = "memory"
    (root / "configs/base.yaml").write_text(
        yaml.safe_dump(cfg_dict, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (root / "configs/features.yaml").write_text(
        (REAL_ROOT / "configs/features.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    cfg = load_config(root / "configs/base.yaml")
    spec = load_feature_spec(root / "configs/features.yaml")
    return root, cfg, spec
