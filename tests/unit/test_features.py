from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vtrain.features.build import FeatureContractError, FeatureSelector


def test_selector_selects_columns() -> None:
    df = pd.DataFrame({"x1": [1.0], "x2": [2.0], "x": [9.0]})
    out = FeatureSelector(["x1", "x2"]).transform(df)
    assert list(out.columns) == ["x1", "x2"]


def test_selector_missing_column_raises() -> None:
    df = pd.DataFrame({"x1": [1.0]})
    with pytest.raises(FeatureContractError):
        FeatureSelector(["x1", "x2"]).transform(df)


def test_selector_missing_value_raises() -> None:
    df = pd.DataFrame({"x1": [1.0, np.nan]})
    with pytest.raises(FeatureContractError):
        FeatureSelector(["x1"]).transform(df)
