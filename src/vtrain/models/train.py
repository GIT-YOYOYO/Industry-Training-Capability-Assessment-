"""训练编排。

流程：拟合基线+候选 -> 验证集选 champion -> 验证集定决策规则 -> 测试集只评估一次
-> 深入评估/解释 -> 训练门禁 -> 报告 -> 模型包 -> 追踪。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score

import vtrain.models  # noqa: F401  触发所有 @register
from vtrain.bundle.bundle import save_bundle
from vtrain.config import AppConfig, FeatureSpec, fingerprint, resolve
from vtrain.data.prepare import Manifest
from vtrain.evaluation.decision import decision_outcome, fit_top_k_rule
from vtrain.evaluation.deep import (
    bootstrap_ci,
    calibration_report,
    error_examples,
    group_metrics,
    quantile_bins,
    slice_gap,
)
from vtrain.evaluation.metrics import classification_metrics, top_k_count, top_k_mask
from vtrain.evaluation.report import write_report
from vtrain.explain.shap_explain import (
    CONTRIBUTION_UNIT,
    explain,
    global_importance,
    raw_score,
)
from vtrain.models.registry import build_model
from vtrain.tracking.tracker import JsonlTracker, MemoryTracker, Tracker, git_info

BASELINES: tuple[str, ...] = ("dummy", "rule_baseline")
ADDITIVITY_TOL = 1e-4


class TrainingGateError(RuntimeError):
    """训练门禁未通过（主指标不如基线，或切片差距超限）。"""


@dataclass(frozen=True)
class TrainingResult:
    champion: str
    model_dir: Path
    report_dir: Path
    report: dict[str, Any]


def _read_splits(proc: Path) -> dict[str, pd.DataFrame]:
    return {
        "train": pd.read_parquet(proc / "train.parquet"),
        "val": pd.read_parquet(proc / "val.parquet"),
        "test": pd.read_parquet(proc / "test.parquet"),
    }


def _make_tracker(cfg: AppConfig, root: Path) -> Tracker:
    if cfg.tracking.backend == "jsonl":
        run_dir = resolve(root, cfg.tracking.dir) / fingerprint(cfg)[:10]
        return JsonlTracker(run_dir)
    return MemoryTracker()


def _additivity_error(est: BaseEstimator, df: pd.DataFrame, features: tuple[str, ...]) -> float:
    sample = df.head(50)
    exps = explain(est, sample, features)
    raw = raw_score(est, sample)
    return max(abs(exp.total - float(raw[i])) for i, exp in enumerate(exps))


def train(cfg: AppConfig, spec: FeatureSpec, root: Path) -> TrainingResult:
    proc = resolve(root, cfg.data.processed_dir)
    data = _read_splits(proc)
    manifest_path = proc / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 manifest，先运行 prepare: {manifest_path}")
    manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    y_col = cfg.data.positive_label
    features = spec.feature_columns
    k_fraction = cfg.decision.k_fraction
    train_df, val_df, test_df = data["train"], data["val"], data["test"]

    candidate_names = tuple(dict.fromkeys(cfg.models.candidates))
    if not candidate_names:
        raise ValueError("configs/base.yaml 的 models.candidates 不能为空")
    all_names = tuple(dict.fromkeys((*BASELINES, *candidate_names)))

    tracker = _make_tracker(cfg, root)
    tracker.set_context(
        {
            **git_info(root),
            "config_fingerprint": fingerprint(cfg),
            "feature_spec_fingerprint": fingerprint(spec),
            "data_fingerprint": manifest.splits["train"].content_sha256,
            "raw_sha256": manifest.raw_sha256,
        }
    )
    tracker.log_params({"models": cfg.models.model_dump(), "decision": cfg.decision.model_dump()})

    fitted: dict[str, BaseEstimator] = {}
    val_metrics: dict[str, dict[str, float]] = {}
    for name in all_names:
        est = build_model(name, cfg, spec)
        est.fit(train_df, train_df[y_col])
        proba = np.asarray(est.predict_proba(val_df)[:, 1], dtype=np.float64)
        fitted[name] = est
        y_val = np.asarray(val_df[y_col], dtype=np.int64)
        val_metrics[name] = classification_metrics(y_val, proba, k_fraction)

    champion_name = max(candidate_names, key=lambda n: val_metrics[n]["pr_auc"])
    champion = fitted[champion_name]

    # 决策规则：只在验证集上定阈值
    val_proba = np.asarray(champion.predict_proba(val_df)[:, 1], dtype=np.float64)
    rule = fit_top_k_rule(val_proba, k_fraction)

    # 测试集：只评估一次
    test_proba = np.asarray(champion.predict_proba(test_df)[:, 1], dtype=np.float64)
    y_test = np.asarray(test_df[y_col], dtype=np.int64)
    test_metrics = classification_metrics(y_test, test_proba, k_fraction)
    test_decisions = top_k_mask(test_proba, top_k_count(len(test_proba), k_fraction)).astype(
        np.int64
    )
    outcome = decision_outcome(y_test, test_decisions)

    # 深入评估（X2）
    ci = bootstrap_ci(
        y_test,
        test_proba,
        average_precision_score,
        cfg.evaluation.bootstrap.n,
        cfg.evaluation.bootstrap.seed,
    )
    baseline_bins = quantile_bins(
        np.asarray(train_df["x"], dtype=np.float64),
        np.asarray(test_df["x"], dtype=np.float64),
        cfg.evaluation.slices.baseline_bins,
    )
    min_size = cfg.evaluation.slices.min_group_size
    gm_baseline = group_metrics(baseline_bins, y_test, test_proba, min_size)
    gm_school = group_metrics(
        np.asarray(test_df["schooltype"], dtype=np.int64), y_test, test_proba, min_size
    )
    cal = calibration_report(y_test, test_proba)
    err = error_examples(test_df, y_test, test_proba, test_decisions, features)

    exp_test = explain(champion, test_df, features)
    gi = global_importance(exp_test)

    # 训练门禁
    gate_metric = cfg.evaluation.slices.gate_metric
    gap_baseline = slice_gap(gm_baseline, gate_metric)
    gap_school = slice_gap(gm_school, gate_metric)
    baseline_pr = val_metrics["dummy"]["pr_auc"]
    delta = val_metrics[champion_name]["pr_auc"] - baseline_pr
    gap_max = max(gap_baseline, gap_school)
    add_err = _additivity_error(champion, test_df, features)
    checks = [
        {
            "name": "主指标超过朴素基线",
            "passed": delta >= cfg.evaluation.acceptance.min_pr_auc_over_baseline,
            "detail": (
                f"ΔPR-AUC={delta:.4f}，门槛 {cfg.evaluation.acceptance.min_pr_auc_over_baseline}"
            ),
        },
        {
            "name": "切片差距在容差内",
            "passed": gap_max <= cfg.evaluation.acceptance.max_slice_gap,
            "detail": (
                f"最大切片差距({gate_metric})={gap_max:.4f}，"
                f"门槛 {cfg.evaluation.acceptance.max_slice_gap}"
            ),
        },
        {
            "name": "解释可加性",
            "passed": add_err <= ADDITIVITY_TOL,
            "detail": f"max|base+Σcontrib-logit|={add_err:.2e}",
        },
    ]
    gate_passed = all(bool(c["passed"]) for c in checks)

    report: dict[str, Any] = {
        "project": cfg.project,
        "champion": champion_name,
        "main_metric": cfg.evaluation.main_metric,
        "label": {
            "column": y_col,
            "threshold": manifest.label_threshold,
            "rule": manifest.label_rule,
        },
        "decision": rule.to_dict(),
        "splits": {key: len(part) for key, part in data.items()},
        "models": {
            name: {"val": metrics, "is_champion": name == champion_name}
            for name, metrics in val_metrics.items()
        },
        "champion_test": test_metrics,
        "decision_test": outcome,
        "bootstrap_test": ci,
        "slices": {
            "gate_metric": gate_metric,
            "baseline": {"groups": gm_baseline, "gap": gap_baseline},
            "schooltype": {"groups": gm_school, "gap": gap_school},
        },
        "calibration_test": cal,
        "errors_test": err,
        "global_importance": [[name, float(value)] for name, value in gi],
        "acceptance": {"passed": gate_passed, "checks": checks},
        "fingerprints": {
            "config": fingerprint(cfg),
            "feature_spec": fingerprint(spec),
            "manifest_train": manifest.splits["train"].content_sha256,
        },
        "git": git_info(root),
    }

    reports_dir = resolve(root, cfg.paths.reports_dir)
    paths = write_report(report, reports_dir)
    tracker.log_artifact(str(paths["json"]))
    tracker.log_artifact(str(paths["markdown"]))

    tracker.log_metrics(
        {f"val_pr_auc_{name}": metrics["pr_auc"] for name, metrics in val_metrics.items()}
        | {
            "test_pr_auc": test_metrics["pr_auc"],
            "test_top_k_recall": outcome["recall"],
            "gate_passed": 1.0 if gate_passed else 0.0,
        }
    )

    if not gate_passed:
        tracker.end()
        failed = [str(c["name"]) for c in checks if not c["passed"]]
        raise TrainingGateError(f"训练门禁未通过：{failed}；诊断报告见 {paths['markdown']}")

    metadata: dict[str, Any] = {
        "model_name": champion_name,
        "feature_columns": list(features),
        "decision_rule": rule.to_dict(),
        "contribution_unit": CONTRIBUTION_UNIT,
        "config_fingerprint": fingerprint(cfg),
        "feature_spec_fingerprint": fingerprint(spec),
        "label_threshold": manifest.label_threshold,
        "label_column": y_col,
        "data_fingerprint": manifest.splits["train"].content_sha256,
        "metrics": {"val": val_metrics[champion_name], "test": test_metrics},
        "training_sizes": {key: len(part) for key, part in data.items()},
    }
    model_dir = resolve(root, cfg.paths.models_dir) / "champion"
    save_bundle(model_dir, champion, metadata)
    tracker.log_artifact(str(model_dir))
    tracker.end()

    return TrainingResult(
        champion=champion_name,
        model_dir=model_dir,
        report_dir=reports_dir,
        report=report,
    )
