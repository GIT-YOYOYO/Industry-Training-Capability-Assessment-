"""数据准备：清洗 -> 分组切分 -> 标签 -> manifest。

口径（写进 docs/design.md 与 ADR）：
- 样本：一名学员一条记录；y/y1..y5 全缺的 10 行剔除。
- 标签：y <= 训练集 y 的 25% 分位 => 不达标(1)。阈值只在训练集上算，val/test 复用同一阈值。
- 切分：按培训中心 schoolid 分组，train/val/test 的中心互不相交（ICC≈0.8，随机切分会泄漏）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import GroupShuffleSplit

from vtrain.config import AppConfig, FeatureSpec, fingerprint, resolve
from vtrain.data.load import load_raw


class CleaningInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    rows_in: int
    rows_dropped: int
    rows_out: int
    missing_by_col: dict[str, int]


class SplitInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    n_rows: int
    n_groups: int
    positive_rate: float
    content_sha256: str


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    project: str
    raw_sha256: str
    config_fingerprint: str
    feature_spec_fingerprint: str
    features: tuple[str, ...]
    excluded: dict[str, str]
    label_source: str
    label_rule: str
    label_threshold: float
    label_column: str
    group_col: str
    group_disjoint: bool
    cleaning: CleaningInfo
    splits: dict[str, SplitInfo]


def frame_fingerprint(df: pd.DataFrame, cols: Sequence[str]) -> str:
    """对选定列做稳定内容指纹（逐行哈希后小端拼接再 sha256）。"""
    sub = df.loc[:, list(cols)].sort_index()
    hashed = pd.util.hash_pandas_object(sub, index=True).astype("uint64").to_numpy()
    return hashlib.sha256(hashed.astype("<u8").tobytes()).hexdigest()


def clean(df: pd.DataFrame, dropna_cols: Sequence[str]) -> tuple[pd.DataFrame, CleaningInfo]:
    """剔除目标侧整行缺失的样本，并记录删了多少、哪些列缺。"""
    rows_in = len(df)
    missing_by_col = {col: int(df[col].isna().sum()) for col in dropna_cols}
    out = df.dropna(subset=list(dropna_cols)).reset_index(drop=True)
    rows_out = len(out)
    info = CleaningInfo(
        rows_in=rows_in,
        rows_dropped=rows_in - rows_out,
        rows_out=rows_out,
        missing_by_col=missing_by_col,
    )
    return out, info


def _split_once(
    df: pd.DataFrame, test_size: float, group_col: str, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df[group_col].to_numpy()
    train_idx, test_idx = next(splitter.split(np.zeros(len(df)), groups=groups))
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def group_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """按 group_col 分组三段切分，中心互不重叠。"""
    trainval_idx, test_idx = _split_once(df, test_size, group_col, seed)
    trainval = df.iloc[trainval_idx].reset_index(drop=True)
    val_frac_of_rest = val_size / (1.0 - test_size)
    train_rel, val_rel = _split_once(trainval, val_frac_of_rest, group_col, seed)
    return {
        "train": trainval.iloc[train_rel].reset_index(drop=True),
        "val": trainval.iloc[val_rel].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def assert_disjoint(splits: Mapping[str, pd.DataFrame], group_col: str) -> bool:
    """强行断言三个切分的分组键互不相交，否则抛错。"""
    names = sorted(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left = set(splits[names[i]][group_col].tolist())
            right = set(splits[names[j]][group_col].tolist())
            overlap = left & right
            if overlap:
                sample = sorted(overlap)[:5]
                raise AssertionError(
                    f"分组切分泄漏：{names[i]} 与 {names[j]} 共享培训中心 {sample}"
                )
    return True


def label_threshold(train: pd.DataFrame, source_col: str, quantile: float) -> float:
    """只在训练集上算标签阈值，避免用到 val/test 的分布。"""
    return float(train[source_col].quantile(quantile))


def add_label(df: pd.DataFrame, source_col: str, threshold: float, column: str) -> pd.DataFrame:
    out = df.copy()
    out[column] = (out[source_col] <= threshold).astype("int8")
    return out


def assert_features_present(df: pd.DataFrame, features: Sequence[str]) -> None:
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise AssertionError(f"缺少特征列: {missing}")
    na_total = int(df.loc[:, list(features)].isna().sum().sum())
    if na_total:
        raise AssertionError(f"特征列存在缺失值共 {na_total} 个，清洗口径需要更新")


def build_manifest(
    cfg: AppConfig,
    spec: FeatureSpec,
    cleaning: CleaningInfo,
    splits: Mapping[str, pd.DataFrame],
    threshold: float,
    disjoint: bool,
) -> Manifest:
    content_cols = (*spec.feature_columns, cfg.data.positive_label)
    infos: dict[str, SplitInfo] = {}
    for name, part in splits.items():
        infos[name] = SplitInfo(
            name=name,
            n_rows=len(part),
            n_groups=int(part[cfg.data.group_col].nunique()),
            positive_rate=float(part[cfg.data.positive_label].mean()),
            content_sha256=frame_fingerprint(part, content_cols),
        )
    rule = (
        f"训练集 {cfg.data.label_source_col} 的 {cfg.data.label_quantile:.0%} 分位"
        f"（阈值={threshold:.6f}）；{cfg.data.label_source_col} <= 阈值 => 1"
    )
    return Manifest(
        project=cfg.project,
        raw_sha256=cfg.data.raw_sha256,
        config_fingerprint=fingerprint(cfg),
        feature_spec_fingerprint=fingerprint(spec),
        features=spec.feature_columns,
        excluded=dict(spec.excluded),
        label_source=cfg.data.label_source_col,
        label_rule=rule,
        label_threshold=threshold,
        label_column=cfg.data.positive_label,
        group_col=cfg.data.group_col,
        group_disjoint=disjoint,
        cleaning=cleaning,
        splits=infos,
    )


def prepare(cfg: AppConfig, spec: FeatureSpec, root: Path) -> Manifest:
    """完整的数据准备编排：读原始 -> 清洗 -> 切分 -> 标签 -> 落盘 + manifest。"""
    raw_path = resolve(root, cfg.data.raw_path)
    df = load_raw(raw_path, cfg.data.sheet, cfg.data.raw_sha256)
    cleaned, cleaning = clean(df, cfg.data.dropna_cols)
    splits = group_split(
        cleaned, cfg.data.group_col, cfg.data.test_size, cfg.data.val_size, cfg.seed
    )
    disjoint = assert_disjoint(splits, cfg.data.group_col)
    threshold = label_threshold(splits["train"], cfg.data.label_source_col, cfg.data.label_quantile)
    splits = {
        name: add_label(part, cfg.data.label_source_col, threshold, cfg.data.positive_label)
        for name, part in splits.items()
    }
    for part in splits.values():
        assert_features_present(part, spec.feature_columns)
    manifest = build_manifest(cfg, spec, cleaning, splits, threshold, disjoint)

    out_dir = resolve(root, cfg.data.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
