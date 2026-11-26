"""一条命令的质量门：ruff lint -> ruff format -> mypy -> pytest(含覆盖率)。

全绿返回 0，任何一步失败返回 1。门槛（覆盖率下限）只写在这一处。
用法：uv run python scripts/ci.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE_MIN = 80
PATHS: list[str] = ["src", "tests", "scripts"]

STEPS: list[tuple[str, list[str]]] = [
    ("ruff lint", ["ruff", "check", *PATHS]),
    ("ruff format", ["ruff", "format", "--check", *PATHS]),
    ("mypy", ["mypy", *PATHS]),
    ("pytest", ["pytest", "-q", "--cov=vtrain", f"--cov-fail-under={COVERAGE_MIN}"]),
]


def run_step(name: str, cmd: list[str]) -> bool:
    print(f"\n=== {name} ===", flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode == 0


def main() -> int:
    for name, cmd in STEPS:
        if not run_step(name, cmd):
            print(f"\n[FAIL] {name}")
            return 1
    print("\n[ALL GREEN] 质量门通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
