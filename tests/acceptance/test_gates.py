"""守门测试（X16）：把规则本身钉住。

任何人想把门槛调低、把特征契约放宽、把标签定义改掉，这些测试都会变红。
合法修改必须同步改测试并写 ADR——这正是“受保护文件”的含义。
"""

from __future__ import annotations

from typing import Any, cast

import yaml

from vtrain.config import project_root


def _base() -> dict[str, Any]:
    data = yaml.safe_load((project_root() / "configs/base.yaml").read_text(encoding="utf-8"))
    return cast("dict[str, Any]", data)


def test_main_metric_gate_not_loosened() -> None:
    base = _base()
    assert base["evaluation"]["acceptance"]["min_pr_auc_over_baseline"] >= 0.05


def test_slice_gap_gate_bounded() -> None:
    base = _base()
    assert base["evaluation"]["acceptance"]["max_slice_gap"] <= 0.25


def test_label_definition_fixed() -> None:
    base = _base()
    assert base["data"]["label_quantile"] == 0.25
    assert base["decision"]["k_fraction"] == 0.1


def test_group_split_by_school() -> None:
    base = _base()
    assert base["data"]["group_col"] == "schoolid"
    assert base["data"]["test_size"] + base["data"]["val_size"] < 1.0


def test_feature_count_fixed() -> None:
    features = yaml.safe_load(
        (project_root() / "configs/features.yaml").read_text(encoding="utf-8")
    )
    assert len(features["groups"]["numeric"]) == 5
