# 自评表

> 每条给出：文件路径、一条验证命令、预期输出。命令均在项目根执行（`uv sync` 之后）。

## 基础模块（满分 60）

### M1 问题定义与设计文档（10）
- 证据：`docs/design.md`（五个问题 + 四口径 + 指标 + “不做什么”）；`docs/adr/` 共 7 条 ADR。
- 验证：`sed -n '1,60p' docs/design.md`；预期看到“四个口径”“不做什么”。
- 一致性：特征口径在 `configs/features.yaml`，代码只从 `spec.feature_columns` 取列。

### M2 仓库骨架与环境（5）
- 证据：`pyproject.toml`、`uv.lock`、`.python-version`（3.13.5）、src 布局、`.gitignore`、`README.md`。
- 验证：`uv sync && uv run python -c "import vtrain; print(vtrain.__version__)"`；预期 `0.1.0`。

### M3 配置与可复现（5）
- 证据：`configs/base.yaml` 单一口径；`src/vtrain/config.py` 用 pydantic `extra="forbid"`；指纹进 manifest 与模型包。
- 验证：`uv run pytest tests/unit/test_config.py -q`；预期全绿（含“拼错键报错”“指纹稳定”）。
- 验证：删 `reports/evaluation.md` 后 `uv run vtrain train` 重跑；预期数字一致（同机、固定种子）。

### M4 数据准备与数据集契约（8）
- 证据：`src/vtrain/data/{schema,load,prepare}.py`；`configs/features.yaml`。
- 验证：`uv run vtrain prepare && cat data/processed/manifest.json`；
  预期 `group_disjoint=true`、`cleaning.rows_dropped=10`、三切分中心数 104/24/32。
- 防泄漏：`assert_disjoint` 强制三切分中心不相交；预处理只在 Pipeline 内、在训练集上拟合。

### M5 训练、基线与评估（9）
- 证据：`src/vtrain/models/{baseline,logreg,lgbm,train}.py`；`reports/evaluation.md`。
- 验证：`uv run vtrain train`；预期 `champion=logreg gate_passed=True`，报告含四种模型对比。
- 测试集只用一次：champion 由验证集 PR-AUC 选出，测试集只在最终评估。

### M6 模型解释（6）
- 证据：`src/vtrain/explain/shap_explain.py`；推理入口返回基准值+贡献。
- 验证：`uv run pytest tests/unit/test_explain.py -q`；预期 logreg 与 lightgbm 都满足“基准值+贡献=原始输出”。
- 局限：单位是对数几率；背景是训练集均值（标准化后为 0）；`docs/design.md` 第 5、6 节。

### M7 模型包与推理入口（7）
- 证据：`src/vtrain/bundle/bundle.py`、`src/vtrain/predict.py`、`src/vtrain/cli.py`。
- 验证：`uv run vtrain predict --input samples.csv`；预期每行含 `proba_low_ability`、`decision`、
  `contributions`，且 `base_value + Σcontributions = logit`。
- 校验：缺列/缺失抛 `FeatureContractError`；特征清单或依赖不符 `load_bundle` 拒绝（见 `tests/unit/test_bundle.py`）。

### M8 测试与质量门（6）
- 证据：`scripts/ci.py`；`tests/{unit,acceptance,e2e}`。
- 验证：`uv run python scripts/ci.py`；预期最后一行 `[ALL GREEN] 质量门通过`，覆盖率 ≥ 80%（实测 91%）。
- 测试不读真实数据：`tests/conftest.py` 现造 24 所学校的小数据。

### M9 与编程智能体协作（4）
- 证据：`AGENTS.md`（规矩，人工撰写）；`docs/ai-log.md`（5 段关键提示词 + 3 个“AI 写错、我发现”的实例）。
- 验证：`sed -n '1,40p' docs/ai-log.md`。

## 扩展项（列出 50 分，按规则封顶计 40）

### X1 多模型对比（10）
- 第一档：同口径同切分下 4 个模型（dummy、rule_baseline、logreg、lightgbm）对比表 → `reports/evaluation.md`。
- 第二档：模型以插件登记进注册表，配置里写名字切换；加模型只加一个文件。
- 证据：`src/vtrain/models/registry.py`；验证：`uv run python -c "from vtrain.models import available_models; print(available_models())"`；
  预期 `('dummy', 'lightgbm', 'logreg', 'rule_baseline')`。

### X2 深入评估（10）
- 四项全做：自助 95% CI、切片评估、概率校准、错误分析 → `reports/evaluation.md` 对应各节。
- 至少一项进自动检查：切片 ROC-AUC 差距超限则训练门禁报错（`src/vtrain/models/train.py` 的 checks）。
- 验证：`sed -n '17,45p' reports/evaluation.md`；预期看到 CI 与三张切片表。

### X4 解释做深（第一档，5）
- 解释进入交付：推理入口对每条预测返回基准值与全部贡献，按可读特征名展示 Top 因素。
- 证据：`uv run vtrain explain --input samples.csv --row 0`；预期 `top_factors` 与 `contributions`。
- 第二档（稳定性/切片解释/人群平均基准）本期未做全，仅认领 5 分。

### X7 实验追踪（10）
- 第一档：每次运行记录参数、指标、工件 → `runs/<指纹>/run.json`。
- 第二档：再记 git sha、是否有未提交改动、配置指纹、数据指纹；追踪经协议适配，测试用内存后端。
- 证据：`src/vtrain/tracking/tracker.py`；验证：`uv run pytest tests/unit/test_tracking.py -q`。

### X15 严格质量门（5）
- mypy strict、import-linter 分层契约、覆盖率门槛三者都进质量门；覆盖率门槛只写在 `pyproject.toml`。
- 证据：`.importlinter`（3 条 forbidden + 1 条 layers 契约，后者天然禁止循环）、`scripts/ci.py`（5 道门）、`pyproject.toml` 的 `tool.coverage.report.fail_under=85`（分支覆盖，实测 88.6%）。
- 验证：`uv run python scripts/ci.py`；预期 5 道门全绿，`import-linter` 显示 `4 kept, 0 broken`。

### X16 守门测试（5）
- 用测试守住规则本身：质量门命令与顺序（`tests/architecture/test_quality_gate.py`）、目录布局与 `py.typed` 与无占位（`tests/architecture/test_layout.py`）、门槛与特征契约（`tests/acceptance/test_gates.py`、`test_feature_contract.py`）。
- 验证：`uv run pytest tests/architecture tests/acceptance -q`；预期全绿；改门槛/混入目标列会变红（见下方破坏实验）。

### X20 三段八步（第一档，5）
- 3 个步骤先提交规格卡与验收测试（打 `spec-guard-*`），看它红，再实现（`step-guard-*`）。
- 证据：`git tag` 有 `spec-guard-1/2/3` 与 `step-guard-1/2/3`；每个 spec 提交的验收测试在实现前跑出红。
- 验证：`git log --oneline --decorate | grep -E "spec-guard|step-guard"` 能看到 spec 在 step 之前。

## 破坏实验记录（M8 要求）

两个实验都用 `_bak` 临时备份、`try/finally` 恢复，恢复后 acceptance 全绿（exit 0，9 passed）。

| 实验 | 改动 | 结果 |
|---|---|---|
| EXP1 | 把 `min_pr_auc_over_baseline` 由 0.05 调低为 0.0 | `test_gates::test_main_metric_gate_not_loosened` **FAILED**，exit 1 |
| EXP2 | 往 `features.yaml` 的 numeric 加 `y`（目标泄漏） | `test_feature_contract::test_forbidden_not_in_features` 与 `test_target_and_components_excluded` **FAILED**，exit 1 |

结论：门槛被悄悄放宽、目标列混入特征，质量门都会变红。

另：`spec-guard-1/2/3` 三个验收测试在实现前分别跑出红（缺 `.importlinter`、覆盖率非单一来源、
缺 `py.typed`），实现后转绿，`step-guard-*` 紧随其后——这是 X20 的“先红后绿”历史证据。

## 复用声明

未复用课程标准仓库或网上项目代码；工程方法（分组切分、质量门、模型包核对、可加性解释）参考讲义与任务书。
所有领域代码、契约、门槛为本项目原创。原始数据：课程提供的 `kaggledata225.xlsx`（sha256 见 README）。