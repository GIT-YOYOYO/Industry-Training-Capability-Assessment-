"""推理入口（函数层）：加载模型包，对一批新数据输出预测、决策、解释。

决策用模型包里保存的验证集等效阈值（单条/小批量没有“批内 Top-K”的语境）。
输入缺列或类型错误时，FeatureSelector 会抛 FeatureContractError，绝不静默填充。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from vtrain.bundle.bundle import load_bundle
from vtrain.evaluation.decision import DecisionRule
from vtrain.explain.shap_explain import explain


def predict_frame(
    pipeline: BaseEstimator,
    metadata: dict[str, Any],
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    features = tuple(metadata["feature_columns"])
    rule = DecisionRule.from_dict(metadata["decision_rule"])
    proba = np.asarray(pipeline.predict_proba(df)[:, 1], dtype=np.float64)
    decisions = (proba >= rule.threshold).astype(np.int64)
    exps = explain(pipeline, df, features)
    rows: list[dict[str, Any]] = []
    for i, exp in enumerate(exps):
        rows.append(
            {
                "row": i,
                "proba_low_ability": float(proba[i]),
                "decision": int(decisions[i]),
                "base_value": exp.base_value,
                "logit": exp.total,
                "top_factors": [
                    {"feature": name, "contribution": value} for name, value in exp.ranked(3)
                ],
                "contributions": exp.contributions,
            }
        )
    return rows


def read_input(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError(f"不支持的文件类型: {suffix}（支持 csv/parquet/xlsx）")


def predict_file(
    model_dir: str | Path,
    input_path: str | Path,
    expected_features: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    pipeline, metadata = load_bundle(model_dir, expected_features)
    return predict_frame(pipeline, metadata, read_input(input_path))
