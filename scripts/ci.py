"""一条命令的质量门：ruff lint -> ruff format -> mypy -> pytest(含覆盖率)。

全绿返回 0，任何一步失败返回 1。门槛（覆盖率下限）只写在这一处。
用法：uv run python scripts/ci.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS: list[str] = ["src", "tests", "scripts"]


def _tool(name: str) -> str:
    """找到虚拟环境里的可执行文件（lint-imports 没有 python -m 入口）。"""
    bindir = Path(sys.executable).parent
    for candidate in (bindir / name, bindir / f"{name}.exe"):
        if candidate.exists():
            return str(candidate)
    return name


def _coverage_floor() -> int:
    """覆盖率门槛只在 pyproject.toml 写一处，这里读它。"""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return int(data["tool"]["coverage"]["report"]["fail_under"])


STEPS: list[tuple[str, list[str]]] = [
    ("ruff lint", ["ruff", "check", *PATHS]),
    ("ruff format", ["ruff", "format", "--check", *PATHS]),
    ("mypy", ["mypy", *PATHS]),
    ("import-linter", [_tool("lint-imports"), "--no-logo"]),
    ("pytest", ["pytest", "-q", "--cov=vtrain", f"--cov-fail-under={_coverage_floor()}"]),
]


def run_step(name: str, cmd: list[str]) -> bool:
    print(f"\n=== {name} ===", flush=True)
    # 子进程统一 UTF-8：中文 Windows 默认 GBK，import-linter 读含中文的源码会解码失败
    env = {**os.environ, "PYTHONUTF8": "1"}
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode == 0


def main() -> int:
    for name, cmd in STEPS:
        if not run_step(name, cmd):
            print(f"\n[FAIL] {name}")
            return 1
    print("\n[ALL GREEN] 质量门通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
