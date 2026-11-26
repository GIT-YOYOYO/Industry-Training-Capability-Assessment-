"""模型包：pipeline（model.joblib）+ 元数据（metadata.json）。

加载时核对：必需元数据齐全、特征清单一致、关键依赖主版本一致。
对不上就拒绝加载并说明原因，而不是“先跑起来再说”。
"""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy
import pandas
import pydantic
import sklearn
from sklearn.base import BaseEstimator

import vtrain

REQUIRED_METADATA: tuple[str, ...] = (
    "model_name",
    "feature_columns",
    "decision_rule",
    "contribution_unit",
    "config_fingerprint",
    "label_threshold",
    "dependency_versions",
)
_MAJOR_CHECKED: tuple[str, ...] = ("scikit-learn", "lightgbm", "pandas", "numpy")


class BundleError(ValueError):
    """模型包缺失、损坏或与当前契约不符。"""


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "pydantic": pydantic.__version__,
        "vtrain": vtrain.__version__,
    }


def save_bundle(
    directory: str | Path,
    pipeline: BaseEstimator,
    metadata: dict[str, Any],
) -> Path:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, target / "model.joblib")
    meta = dict(metadata)
    meta["dependency_versions"] = dependency_versions()
    (target / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def _check_dependency_majors(saved: dict[str, str]) -> None:
    current = dependency_versions()
    for name in _MAJOR_CHECKED:
        if name not in saved:
            continue
        saved_major = saved[name].split(".")[0]
        current_major = current[name].split(".")[0]
        if saved_major != current_major:
            raise BundleError(
                f"依赖主版本不符，拒绝加载：{name} 包={saved[name]} 当前={current[name]}"
            )


def load_bundle(
    directory: str | Path,
    expected_features: Sequence[str] | None = None,
) -> tuple[BaseEstimator, dict[str, Any]]:
    target = Path(directory)
    model_path = target / "model.joblib"
    meta_path = target / "metadata.json"
    if not model_path.is_file() or not meta_path.is_file():
        raise BundleError(f"模型包不完整（缺 model.joblib 或 metadata.json）: {target}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_METADATA if key not in meta]
    if missing:
        raise BundleError(f"模型包元数据缺少字段: {missing}")
    if expected_features is not None:
        saved_features = tuple(meta["feature_columns"])
        if saved_features != tuple(expected_features):
            raise BundleError(
                f"特征清单不符：包={list(saved_features)}，期望={list(expected_features)}"
            )
    _check_dependency_majors(dict(meta["dependency_versions"]))
    pipeline: BaseEstimator = joblib.load(model_path)
    return pipeline, meta
