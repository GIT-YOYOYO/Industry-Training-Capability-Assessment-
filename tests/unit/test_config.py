from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vtrain.config import fingerprint, load_config, load_feature_spec, project_root


def test_load_config_ok() -> None:
    root = project_root()
    cfg = load_config(root / "configs/base.yaml")
    assert cfg.seed > 0
    assert cfg.features.spec_path


def test_typo_key_rejected(tmp_path: Path) -> None:
    root = project_root()
    raw = yaml.safe_load((root / "configs/base.yaml").read_text(encoding="utf-8"))
    raw["data"]["test_szie"] = 0.1
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_fingerprint_stable() -> None:
    root = project_root()
    a = load_config(root / "configs/base.yaml")
    b = load_config(root / "configs/base.yaml")
    assert fingerprint(a) == fingerprint(b)
    assert len(fingerprint(a)) == 64


def test_feature_spec_columns() -> None:
    root = project_root()
    cfg = load_config(root / "configs/base.yaml")
    spec = load_feature_spec(root / cfg.features.spec_path)
    assert spec.feature_columns == ("x1", "x2", "x3", "x4", "x5")
