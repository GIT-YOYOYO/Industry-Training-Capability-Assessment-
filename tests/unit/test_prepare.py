from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vtrain.data.prepare import (
    add_label,
    assert_disjoint,
    clean,
    frame_fingerprint,
    group_split,
    label_threshold,
)


def test_clean_drops_missing() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [1, 2, 3]})
    _out, info = clean(df, ["a"])
    assert info.rows_in == 3
    assert info.rows_dropped == 1
    assert info.rows_out == 2
    assert info.missing_by_col["a"] == 1


def test_group_split_disjoint() -> None:
    df = pd.DataFrame({"g": np.repeat(np.arange(20), 10), "v": np.arange(200)})
    splits = group_split(df, "g", 0.25, 0.25, seed=0)
    assert assert_disjoint(splits, "g")
    assert set(splits["train"]["g"]).isdisjoint(set(splits["test"]["g"]))
    assert set(splits["val"]["g"]).isdisjoint(set(splits["test"]["g"]))


def test_assert_disjoint_raises_on_overlap() -> None:
    a = pd.DataFrame({"g": [1, 2]})
    b = pd.DataFrame({"g": [2, 3]})
    with pytest.raises(AssertionError):
        assert_disjoint({"train": a, "test": b}, "g")


def test_label_threshold_and_add() -> None:
    tr = pd.DataFrame({"y": np.arange(100.0)})
    thr = label_threshold(tr, "y", 0.25)
    out = add_label(tr, "y", thr, "low")
    assert out["low"].mean() == pytest.approx(0.25, abs=0.02)
    assert set(out["low"].unique()) <= {0, 1}


def test_frame_fingerprint_stable() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    assert frame_fingerprint(df, ["a", "b"]) == frame_fingerprint(df.copy(), ["a", "b"])
    assert frame_fingerprint(df, ["a"]) != frame_fingerprint(df, ["a", "b"])
