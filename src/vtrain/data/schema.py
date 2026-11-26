"""原始数据的列契约：进模型前后都靠它守住“哪些列应该存在”。"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

# 原始 xlsx 的 15 列，顺序即来源顺序
RAW_COLUMNS: tuple[str, ...] = (
    "stuid",
    "schoolid",
    "schooltype",
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x",
    "y1",
    "y2",
    "y3",
    "y4",
    "y5",
    "y",
)


class SchemaError(ValueError):
    """数据列与契约不符。"""


def validate_raw_columns(df: pd.DataFrame, expected: Sequence[str] = RAW_COLUMNS) -> None:
    """缺列或多列都报错，并指出具体是哪几列。"""
    actual = set(df.columns)
    want = set(expected)
    missing = sorted(want - actual)
    extra = sorted(actual - want)
    if missing or extra:
        raise SchemaError(f"数据列与契约不符：缺少={missing}，多出={extra}")
