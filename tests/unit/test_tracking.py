from __future__ import annotations

import json
from pathlib import Path

from vtrain.tracking.tracker import JsonlTracker, MemoryTracker, git_info


def test_memory_tracker() -> None:
    tracker = MemoryTracker()
    tracker.set_context({"git_sha": "x"})
    tracker.log_params({"p": 1})
    tracker.log_metrics({"m": 0.5})
    tracker.log_artifact("path")
    tracker.end()
    assert tracker.metrics["m"] == 0.5
    assert tracker.artifacts == ["path"]
    assert tracker.ended


def test_jsonl_tracker(tmp_path: Path) -> None:
    tracker = JsonlTracker(tmp_path)
    tracker.log_params({"epochs": 3})
    tracker.log_metrics({"pr_auc": 0.7})
    tracker.end()
    data = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert data["params"]["epochs"] == 3
    assert data["metrics"]["pr_auc"] == 0.7


def test_git_info_shape() -> None:
    info = git_info(".")
    assert set(info) == {"git_sha", "git_dirty"}
