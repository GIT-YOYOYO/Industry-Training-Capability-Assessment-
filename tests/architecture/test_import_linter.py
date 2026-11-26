"""验收：分层契约（X15）。.importlinter 必须存在，且质量门必须真的跑 lint-imports。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_importlinter_config_exists() -> None:
    assert (ROOT / ".importlinter").is_file(), "缺少 .importlinter 分层契约配置"


def test_importlinter_declares_forbidden_contract() -> None:
    text = (ROOT / ".importlinter").read_text(encoding="utf-8")
    assert "importlinter:contract" in text, "没有声明任何契约"
    assert "forbidden" in text, "至少要有 forbidden 契约来禁止反向依赖"


def test_ci_runs_import_linter() -> None:
    ci = (ROOT / "scripts" / "ci.py").read_text(encoding="utf-8")
    assert "lint-imports" in ci or "importlinter" in ci, "质量门没有跑 import-linter"
