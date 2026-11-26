"""命令行入口：prepare / train / predict / explain。

所有命令都从项目根的 configs/base.yaml 取口径；路径参数默认相对项目根。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from vtrain.bundle.bundle import load_bundle
from vtrain.config import (
    AppConfig,
    FeatureSpec,
    load_config,
    load_feature_spec,
    project_root,
    resolve,
)
from vtrain.data.prepare import prepare
from vtrain.models.train import train
from vtrain.predict import predict_frame, read_input


def _load_cfg(root: Path) -> tuple[AppConfig, FeatureSpec]:
    cfg = load_config(resolve(root, "configs/base.yaml"))
    spec = load_feature_spec(resolve(root, cfg.features.spec_path))
    return cfg, spec


def _resolve_model(root: Path, model: str) -> Path:
    p = Path(model)
    return p if p.is_absolute() else resolve(root, model)


def cmd_prepare(args: argparse.Namespace) -> int:
    root = project_root()
    cfg, spec = _load_cfg(root)
    manifest = prepare(cfg, spec, root)
    print(json.dumps(json.loads(manifest.model_dump_json()), ensure_ascii=False, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    root = project_root()
    cfg, spec = _load_cfg(root)
    result = train(cfg, spec, root)
    gate = bool(result.report["acceptance"]["passed"])
    print(f"champion={result.champion} gate_passed={gate}")
    print(f"report={result.report_dir / 'evaluation.md'}")
    print(f"model={result.model_dir}")
    return 0


def _run_predict(
    root: Path,
    model: str,
    input_path: str,
    row: int | None = None,
) -> list[dict[str, Any]]:
    pipeline, metadata = load_bundle(_resolve_model(root, model))
    df = read_input(input_path)
    if row is not None:
        if not 0 <= row < len(df):
            raise SystemExit(f"--row {row} 超出范围（共 {len(df)} 行）")
        df = df.iloc[[row]]
    return predict_frame(pipeline, metadata, df)


def cmd_predict(args: argparse.Namespace) -> int:
    rows = _run_predict(project_root(), args.model, args.input)
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"已写入 {args.output}（{len(rows)} 条）")
    else:
        print(text)
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    rows = _run_predict(project_root(), args.model, args.input, row=args.row)
    print(json.dumps(rows[0], ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vtrain", description="职业技术培训能力不达标风险筛选")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="数据准备：清洗、分组切分、标签、manifest")
    p_prep.set_defaults(func=cmd_prepare)

    p_train = sub.add_parser("train", help="训练、评估、写报告与模型包")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="对一批新数据输出预测、决策、解释")
    p_pred.add_argument("--input", required=True, help="csv/parquet/xlsx 输入文件")
    p_pred.add_argument("--model", default="models/champion", help="模型包目录")
    p_pred.add_argument("--output", default=None, help="结果 JSON 路径；缺省打印到标准输出")
    p_pred.set_defaults(func=cmd_predict)

    p_exp = sub.add_parser("explain", help="对单条样本输出详细解释")
    p_exp.add_argument("--input", required=True)
    p_exp.add_argument("--row", type=int, default=0)
    p_exp.add_argument("--model", default="models/champion")
    p_exp.set_defaults(func=cmd_explain)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Windows 控制台默认 GBK，报告里有 ²、中文等字符，统一切到 UTF-8
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
