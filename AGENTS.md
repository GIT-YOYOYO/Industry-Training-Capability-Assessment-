# AGENTS.md — 给智能体和新同事的规矩

本文件由项目负责人撰写，不交给 AI 生成。

## 这是什么项目

职业技术培训能力不达标风险筛选：用培训前的 5 个基线测评维度（x1–x5）预测学员培训后是否
“能力不达标”，用于名额有限的强化辅导班筛选。仓库是可安装的 Python 包（src 布局），
配置在 `configs/`，口径数字只写一处，质量门是一条命令。

## 怎么算完成

`uv run python scripts/ci.py` 全绿（ruff lint + ruff format + mypy + pytest，覆盖率 ≥ 80%）。

## 绝对不要做的事

- 不要把 `y1–y5`、`y`、`x`、`stuid`、`schoolid`、`schooltype` 加进特征。禁用理由见
  `configs/features.yaml`；改特征必须同步 `tests/acceptance/test_feature_contract.py`。
- 不要降低 `configs/base.yaml` 里的门槛、加 skip/xfail、成片 noqa 来“让它通过”。真要改门槛，
  先写 ADR，再同步 `tests/acceptance/test_gates.py`（这些是受保护文件）。
- 不要在测试里读真实数据或连任何外部服务；测试只用量产的小数据（`tests/conftest.py`）。
- 不要手改 `reports/` 里的数字；报告必须由 `uv run vtrain train` 生成。
- 不要提交 `data/`、`models/`、`runs/`、密钥或个人身份信息。

## 仓库约定

- 所有口径数字（比例、种子、阈值、超参、路径、门槛）只改 `configs/*.yaml`，代码里不写魔法数字。
- 新增模型：在 `src/vtrain/models/` 加一个文件并 `@register`，调用方不改。
- 新增特征口径、改标签定义、改切分方式：都先写 ADR（`docs/adr/`）。
- 提交信息用“动词: 说明”，里程碑打 `step-*` tag；先写规格卡和验收测试的步骤打 `spec-*` tag。
- 解释必须满足“基准值 + 全部贡献 = 模型原始输出”，由 `tests/unit/test_explain.py` 守着。