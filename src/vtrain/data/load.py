"""读取原始数据并核对指纹。

原始数据不进 Git；它的 sha256 写在 configs/base.yaml，加载时逐位核对。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from vtrain.data.schema import validate_raw_columns


class DataIntegrityError(ValueError):
    """原始文件指纹与配置记录不一致。"""


def sha256_of_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(
    path: str | Path,
    sheet: str,
    expected_sha256: str | None = None,
) -> pd.DataFrame:
    """读取 Excel，校验列契约，并在给了指纹时核对 sha256。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"找不到原始数据: {p}。请把 kaggledata225.xlsx 放到 data/raw/，详见 README。"
        )
    if expected_sha256 is not None:
        actual = sha256_of_file(p)
        if actual != expected_sha256:
            raise DataIntegrityError(
                f"原始数据指纹不符：期望 {expected_sha256}，实际 {actual}。"
                "换数据时请同步修改 configs/base.yaml 的 raw_sha256 并说明原因。"
            )
    df = pd.read_excel(p, sheet_name=sheet)
    validate_raw_columns(df)
    return df
