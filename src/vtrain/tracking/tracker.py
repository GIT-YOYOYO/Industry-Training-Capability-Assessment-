"""实验追踪：一个协议 + JSONL 后端 + 内存后端。

- 协议让调用方只依赖 log_* 接口，换后端（MLflow 等）不改调用方（X7 第二档）。
- 每次运行记录参数、指标、工件，外加 git sha、是否有未提交改动、配置指纹、数据指纹。
- 测试用内存后端，不连任何真实服务。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Protocol


class Tracker(Protocol):
    def set_context(self, context: dict[str, Any]) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float]) -> None: ...
    def log_artifact(self, path: str) -> None: ...
    def end(self) -> None: ...


class MemoryTracker:
    def __init__(self) -> None:
        self.context: dict[str, Any] = {}
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.artifacts: list[str] = []
        self.ended = False

    def set_context(self, context: dict[str, Any]) -> None:
        self.context.update(context)

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)

    def end(self) -> None:
        self.ended = True


class JsonlTracker:
    """把一次运行写成一个 run.json（便于人读，也便于按行追加历史）。"""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.record: dict[str, Any] = {
            "context": {},
            "params": {},
            "metrics": {},
            "artifacts": [],
        }
        self.ended = False

    def set_context(self, context: dict[str, Any]) -> None:
        self.record["context"].update(context)
        self._flush()

    def log_params(self, params: dict[str, Any]) -> None:
        self.record["params"].update(params)
        self._flush()

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.record["metrics"].update(metrics)
        self._flush()

    def log_artifact(self, path: str) -> None:
        self.record["artifacts"].append(path)
        self._flush()

    def end(self) -> None:
        self.ended = True
        self._flush()

    def _flush(self) -> None:
        (self.run_dir / "run.json").write_text(
            json.dumps(self.record, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def git_info(root: str | Path) -> dict[str, Any]:
    """记录 git sha 与是否有未提交改动；不在仓库里时返回 None（不抛错）。"""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"git_sha": sha, "git_dirty": bool(porcelain.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"git_sha": None, "git_dirty": None}
