"""配置加载与校验。

设计要点：
- 所有口径数字只在 configs/*.yaml 写一处；代码从 AppConfig 取，不写魔法数字。
- pydantic 严格模式（extra="forbid"）：拼错的键、错误的类型当场报错，不静默取默认值。
- 配置算出 sha256 指纹，写进 manifest 与模型包，用于“同一份配置”的核对。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid", frozen=True)

ModelName = Literal["logreg", "lightgbm"]


class BootstrapConfig(BaseModel):
    model_config = _STRICT
    n: int = Field(gt=0)
    seed: int


class SliceConfig(BaseModel):
    model_config = _STRICT
    baseline_bins: int = Field(ge=2)
    min_group_size: int = Field(gt=0)
    gate_metric: Literal["roc_auc", "pr_auc"]


class AcceptanceConfig(BaseModel):
    model_config = _STRICT
    max_slice_gap: float = Field(ge=0.0, le=1.0)
    min_pr_auc_over_baseline: float


class EvaluationConfig(BaseModel):
    model_config = _STRICT
    main_metric: Literal["pr_auc", "roc_auc"]
    bootstrap: BootstrapConfig
    slices: SliceConfig
    acceptance: AcceptanceConfig


class DataConfig(BaseModel):
    model_config = _STRICT
    raw_path: str
    raw_sha256: str
    sheet: str
    processed_dir: str
    test_size: float = Field(gt=0.0, lt=1.0)
    val_size: float = Field(gt=0.0, lt=1.0)
    group_col: str
    label_source_col: str
    label_quantile: float = Field(gt=0.0, lt=1.0)
    positive_label: str
    dropna_cols: tuple[str, ...]


class FeaturesConfig(BaseModel):
    model_config = _STRICT
    spec_path: str


class LogRegParams(BaseModel):
    model_config = _STRICT
    C: float = Field(gt=0.0)
    max_iter: int = Field(gt=0)
    class_weight: Literal["balanced"] | None = None


class LgbmParams(BaseModel):
    model_config = _STRICT
    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    num_leaves: int = Field(gt=1)
    min_child_samples: int = Field(ge=1)
    subsample: float = Field(gt=0.0, le=1.0)
    subsample_freq: int = Field(ge=1)
    colsample_bytree: float = Field(gt=0.0, le=1.0)
    reg_lambda: float = Field(ge=0.0)
    n_jobs: int


class ModelsConfig(BaseModel):
    model_config = _STRICT
    candidates: tuple[ModelName, ...]
    selection: Literal["best_by_val_pr_auc"]
    logreg: LogRegParams
    lightgbm: LgbmParams


class DecisionConfig(BaseModel):
    model_config = _STRICT
    strategy: Literal["top_k", "threshold"]
    k_fraction: float = Field(gt=0.0, lt=1.0)


class TrackingConfig(BaseModel):
    model_config = _STRICT
    backend: Literal["jsonl", "memory"]
    dir: str


class PathsConfig(BaseModel):
    model_config = _STRICT
    models_dir: str
    reports_dir: str


class AppConfig(BaseModel):
    model_config = _STRICT
    project: str
    seed: int
    data: DataConfig
    features: FeaturesConfig
    models: ModelsConfig
    decision: DecisionConfig
    evaluation: EvaluationConfig
    tracking: TrackingConfig
    paths: PathsConfig


class FeatureGroupConfig(BaseModel):
    model_config = _STRICT
    numeric: tuple[str, ...]


class FeatureSpec(BaseModel):
    model_config = _STRICT
    groups: FeatureGroupConfig
    excluded: dict[str, str]

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.groups.numeric


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件顶层必须是映射: {path}")
    return cast("dict[str, Any]", raw)


def load_config(path: str | Path) -> AppConfig:
    """加载并严格校验主配置。拼错的键会抛 pydantic.ValidationError。"""
    return AppConfig.model_validate(_read_yaml(Path(path)))


def load_feature_spec(path: str | Path) -> FeatureSpec:
    return FeatureSpec.model_validate(_read_yaml(Path(path)))


def fingerprint(model: BaseModel) -> str:
    """配置/对象的稳定指纹：键排序后 JSON 的 sha256。"""
    payload = json.dumps(model.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def project_root() -> Path:
    """从本文件向上找到含 pyproject.toml 的目录。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def resolve(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p
