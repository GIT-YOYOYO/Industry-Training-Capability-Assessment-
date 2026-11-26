"""验收：特征契约。禁用列绝不能进模型，每条禁用都要有理由。"""

from __future__ import annotations

from vtrain.config import FeatureSpec, load_config, load_feature_spec, project_root
from vtrain.data.schema import RAW_COLUMNS


def _spec() -> FeatureSpec:
    root = project_root()
    cfg = load_config(root / "configs/base.yaml")
    return load_feature_spec(root / cfg.features.spec_path)


def test_forbidden_not_in_features() -> None:
    spec = _spec()
    assert set(spec.excluded).isdisjoint(spec.feature_columns)


def test_every_excluded_has_reason() -> None:
    spec = _spec()
    for col, reason in spec.excluded.items():
        assert reason.strip(), f"{col} 缺少禁用理由"


def test_features_are_raw_columns() -> None:
    spec = _spec()
    assert set(spec.feature_columns) <= set(RAW_COLUMNS)


def test_target_and_components_excluded() -> None:
    spec = _spec()
    for col in ("y", "y1", "y2", "y3", "y4", "y5"):
        assert col not in spec.feature_columns
        assert col in spec.excluded
