"""把评估结果渲染成 JSON + 一页 Markdown。所有数字都来自代码，报告不手填。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _f(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _metric_table(models: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| 模型 | 验证 PR-AUC | 验证 ROC-AUC | 验证 Top-K 召回 | 是否入选 |",
        "|---|---|---|---|---|",
    ]
    for name, info in models.items():
        val = info["val"]
        chosen = "✓" if info.get("is_champion") else ""
        lines.append(
            f"| {name} | {_f(val['pr_auc'])} | {_f(val['roc_auc'])} | "
            f"{_f(val['top_k_recall'])} | {chosen} |"
        )
    return lines


def _slice_table(groups: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| 分组 | 样本量 | 正例率 | PR-AUC | ROC-AUC | lift(PR/正例率) |",
        "|---|---|---|---|---|---|",
    ]
    for key, info in groups.items():
        lines.append(
            f"| {key} | {int(info['n'])} | {_f(info['positive_rate'])} | "
            f"{_f(info['pr_auc'])} | {_f(info['roc_auc'])} | {info['lift']:.2f} |"
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    r = report
    lines: list[str] = []
    lines.append(f"# 评估报告：{r['project']}")
    lines.append("")
    lines.append(f"- 入选模型（champion）：**{r['champion']}**")
    lines.append(f"- 主指标：**{r['main_metric']}**（验证集选拔，测试集只评估一次）")
    lines.append(f"- 标签：{r['label']['rule']}")
    lines.append(
        f"- 决策规则：{r['decision']['strategy']}，名额比例 {r['decision']['k_fraction']:.0%}，"
        f"阈值 {r['decision']['threshold']:.4f}"
    )
    lines.append(
        f"- 切分规模：train={r['splits']['train']}，"
        f"val={r['splits']['val']}，test={r['splits']['test']}"
    )
    lines.append("")
    lines.append("## 模型对比（验证集）")
    lines.extend(_metric_table(r["models"]))
    lines.append("")
    lines.append("## 入选模型测试集表现")
    test = r["champion_test"]
    lines.append(
        f"- PR-AUC {_f(test['pr_auc'])} ｜ ROC-AUC {_f(test['roc_auc'])} ｜ "
        f"Brier {_f(test['brier'])} ｜ 正例率 {_f(test['positive_rate'])}"
    )
    ci = r["bootstrap_test"]
    lines.append(
        f"- PR-AUC 自助 95% 区间：[{_f(ci['low'])}, {_f(ci['high'])}]（均值 {_f(ci['mean'])}）"
    )
    dec = r["decision_test"]
    lines.append(
        f"- 名额制：选中比例 {_f(dec['selected_fraction'])}，召回 {_f(dec['recall'])}，"
        f"精确率 {_f(dec['precision'])}"
    )
    lines.append("")
    lines.append("## 切片评估")
    gate_metric = r["slices"].get("gate_metric", "roc_auc")
    lines.append(
        "> PR-AUC 的下界是组内正例率，跨组不可直接比较；"
        f"切片门禁与比较以 {gate_metric} 为准，lift = PR-AUC / 组内正例率。"
    )
    for key in ("baseline", "schooltype"):
        block = r["slices"][key]
        lines.append(f"### 按 {key} 分组（组间 {gate_metric} 差距 {_f(block['gap'])}）")
        lines.extend(_slice_table(block["groups"]))
        lines.append("")
    lines.append("## 概率校准（测试集）")
    cal = r["calibration_test"]
    lines.append(f"- Brier {_f(cal['brier'])}，ECE {_f(cal['ece'])}")
    lines.append("")
    lines.append("## 错误分析（测试集）")
    err = r["errors_test"]
    lines.append(
        f"- 漏报（实际不达标、未入选）：{err['n_false_negative']}；"
        f"误报（实际达标、被选中）：{err['n_false_positive']}"
    )
    lines.append("")
    lines.append("## 全局解释（平均 |贡献| 排序，对数几率单位）")
    for name, value in r["global_importance"]:
        lines.append(f"- {name}: {_f(value)}")
    lines.append("")
    lines.append("## 训练门禁")
    for chk in r["acceptance"]["checks"]:
        mark = "✓" if chk["passed"] else "✗"
        lines.append(f"- {mark} {chk['name']}：{chk['detail']}")
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "evaluation.json"
    md_path = target / "evaluation.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
