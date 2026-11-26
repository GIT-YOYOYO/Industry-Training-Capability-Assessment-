"""验收：质量门元测试（X16）。守门的命令、顺序、覆盖率门槛来源都不能被悄悄改。"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load_ci() -> Any:
    spec = importlib.util.spec_from_file_location("vtrain_ci_under_test", ROOT / "scripts" / "ci.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quality_gate_has_five_steps() -> None:
    ci = _load_ci()
    assert len(ci.STEPS) == 5


def test_quality_gate_order() -> None:
    ci = _load_ci()
    names = [name for name, _ in ci.STEPS]
    assert names == ["ruff lint", "ruff format", "mypy", "import-linter", "pytest"]


def test_coverage_floor_reads_pyproject() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floor = data["tool"]["coverage"]["report"]["fail_under"]
    assert floor >= 80, "覆盖率门槛不得低于 80"
    ci = _load_ci()
    pytest_cmd = dict(ci.STEPS)["pytest"]
    assert pytest_cmd[-1] == f"--cov-fail-under={floor}"


def test_no_hardcoded_coverage_constant() -> None:
    text = (ROOT / "scripts" / "ci.py").read_text(encoding="utf-8")
    assert "COVERAGE_MIN" not in text, "覆盖率门槛应从 pyproject 读，不能在 ci.py 写死"