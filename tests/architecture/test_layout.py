"""验收：布局与无占位（X16）。目录、类型标记、占位符都被测试钉住。"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "vtrain"


def test_top_level_layout() -> None:
    for rel in (
        "README.md",
        "AGENTS.md",
        "pyproject.toml",
        "uv.lock",
        "configs/base.yaml",
        "configs/features.yaml",
        ".importlinter",
        "scripts/ci.py",
    ):
        assert (ROOT / rel).is_file(), f"缺少 {rel}"


def test_tests_layout() -> None:
    for name in ("unit", "acceptance", "e2e", "architecture"):
        assert (ROOT / "tests" / name).is_dir(), f"缺少 tests/{name}"


def test_src_packages() -> None:
    for name in ("data", "features", "models", "evaluation", "explain", "bundle", "tracking"):
        assert (SRC / name).is_dir(), f"缺少 src/vtrain/{name}"


def test_py_typed_marker() -> None:
    assert (SRC / "py.typed").is_file(), "缺少 py.typed（类型包标记）"


def test_no_placeholder_markers() -> None:
    bad = re.compile(r"\b(TODO|FIXME|XXX|NotImplementedError)\b")
    hits = []
    for path in SRC.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if bad.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not hits, "发现占位标记:\n" + "\n".join(hits)


def test_type_ignore_has_error_code() -> None:
    bare = re.compile(r"#\s*type:\s*ignore(?!\[)")
    hits = []
    for path in SRC.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if bare.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert not hits, "存在不带错误码的 type: ignore:\n" + "\n".join(hits)