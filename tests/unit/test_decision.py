from __future__ import annotations

import pytest

from vtrain.evaluation.decision import (
    DecisionRule,
    apply_rule,
    decision_outcome,
    fit_top_k_rule,
)


def test_fit_top_k_rule() -> None:
    rule = fit_top_k_rule([0.1, 0.2, 0.3, 0.4], 0.5)
    assert rule.strategy == "top_k"
    assert rule.k_fraction == 0.5
    assert rule.threshold == pytest.approx(0.3)


def test_apply_rule() -> None:
    rule = DecisionRule(strategy="top_k", k_fraction=0.5, threshold=0.3)
    decisions = apply_rule([0.1, 0.35, 0.5], rule)
    assert decisions.tolist() == [0, 1, 1]


def test_rule_roundtrip_dict() -> None:
    rule = DecisionRule(strategy="top_k", k_fraction=0.1, threshold=0.42)
    assert DecisionRule.from_dict(rule.to_dict()) == rule


def test_decision_outcome() -> None:
    out = decision_outcome([1, 0, 1, 0], [1, 0, 1, 1])
    assert out["n_selected"] == 3
    assert out["recall"] == 1.0
    assert out["precision"] == pytest.approx(2 / 3)
