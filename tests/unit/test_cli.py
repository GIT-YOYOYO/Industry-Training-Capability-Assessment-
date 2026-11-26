"""只测参数解析层，不执行会读真实数据的命令。"""

from __future__ import annotations

from vtrain.cli import build_parser


def test_parser_has_all_commands() -> None:
    parser = build_parser()
    simple = parser.parse_args(["prepare"])
    assert simple.command == "prepare"
    train = parser.parse_args(["train"])
    assert train.command == "train"
    predict = parser.parse_args(["predict", "--input", "a.csv"])
    assert predict.command == "predict"
    explain = parser.parse_args(["explain", "--input", "a.csv"])
    assert explain.command == "explain"


def test_predict_defaults() -> None:
    args = build_parser().parse_args(["predict", "--input", "a.csv"])
    assert args.model == "models/champion"
    assert args.output is None


def test_explain_row_is_int() -> None:
    args = build_parser().parse_args(["explain", "--input", "a.csv", "--row", "2"])
    assert args.row == 2
