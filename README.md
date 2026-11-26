# 职业技术培训能力不达标风险筛选

在学员参加培训**之前**，用 5 个维度的基线测评（x1–x5）预测其培训后是否“能力不达标”，
用于名额有限的强化辅导班筛选。项目的重点不是模型分数，而是：口径清楚、可复现、可解释、有质量门。

## 关键结论

- 入选模型：**逻辑回归**（验证集 PR-AUC 0.681）；测试集 PR-AUC **0.733**，自助 95% CI **[0.711, 0.757]**。
- 名额制（取风险最高 10%）：召回 **30.4%**、精确率 **84.3%**（随机抓 10% 只能召回约 10%，提升约 3 倍）。
- 复杂模型没有赢：LightGBM 与逻辑回归持平略低，说明信号近似线性；主要价值在可解释与可部署。

## 数据

- 文件：`data/raw/kaggledata225.xlsx`（原始数据不进 Git，见 `.gitignore`）。
- sha256：`ce96c556f88daea9d3157b89eadfee44996adc712a55f527e98e4747e02e4129`。
- 规模：32,165 行 × 15 列；剔除 y 侧缺失 10 行后 32,155 行，160 个培训中心。
- 来源：课程提供的 `kaggledata225.xlsx`。若从别处获取，请先核对 sha256。

## 环境与复现

需要 Python 3.12 与 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync                                       # 按 uv.lock 锁定安装依赖
# 把 kaggledata225.xlsx 放到 data/raw/
uv run vtrain prepare                         # 清洗 -> 分组切分 -> 标签 -> manifest
uv run vtrain train                           # 训练/评估 -> reports/ 与 models/champion
uv run vtrain predict --input samples.csv     # 推理：预测 + 决策 + 解释
uv run python scripts/ci.py                   # 质量门：ruff + format + mypy + pytest(覆盖率>=80%)
```

预期输出：

- `vtrain train` 末尾打印 `champion=logreg gate_passed=True`。
- `scripts/ci.py` 末尾打印 `[ALL GREEN] 质量门通过`，覆盖率 >= 80%。
- `reports/evaluation.md` 与 `reports/evaluation.json` 由代码生成，重跑内容一致（测试集按固定种子）。

`data/processed/` 下的 `train/val/test.parquet` 与 `manifest.json` 由 `prepare` 生成；
`reports/` 与 `models/champion/` 由 `train` 生成。

## 目录结构

```
.
├── pyproject.toml  uv.lock  .python-version
├── README.md  AGENTS.md
├── configs/
│   ├── base.yaml            全部口径数字（切分、种子、阈值、超参、门槛）
│   └── features.yaml        特征清单：x1-x5 入模，其余列禁用并写理由
├── src/vtrain/
│   ├── config.py            严格加载配置 + 指纹
│   ├── data/                契约、加载、清洗/切分/标签/manifest
│   ├── features/            特征选择器（训练/推理共用）
│   ├── models/              注册表 + 两个基线 + 逻辑回归 + LightGBM + 训练编排
│   ├── evaluation/          指标、决策规则、深入评估、报告
│   ├── explain/             线性 SHAP / TreeSHAP 可加性解释
│   ├── bundle/              模型包读写与核对
│   ├── tracking/            追踪协议 + JSONL/内存后端
│   ├── predict.py           推理函数
│   └── cli.py               prepare / train / predict / explain
├── tests/{unit,acceptance,e2e}/
├── scripts/ci.py            一条命令的质量门
├── docs/{design.md,adr/,specs/,ai-log.md,self-assessment.md}
├── reports/                 评估报告（代码生成）
├── data/  models/  runs/    不进 Git
└── samples.csv              推理示例输入
```

## 设计要点（详见 `docs/design.md`）

- **四口径**：样本、标签、时点与切分、决策，全部写死并可核对。
- **二分类**：`low_ability = 1 iff 培训后综合能力 y <= 训练集 25% 分位`；主指标 PR-AUC。
- **分组切分**：按培训中心 `schoolid` 分组，中心互不相交（ICC≈0.8，随机切分会泄漏）。
- **特征**：只用 x1–x5；禁用 `x`（x1–x5 的确定性线性组合，冗余）、`y1–y5`/`y`（目标泄漏）、
  标识符、`schooltype`（公平性）。
- **解释**：基准值 + 全部贡献 = 模型原始输出（对数几率），有测试守着。
- **质量门**：主指标须超基线、切片差距须在容差内、解释须可加回；任一不过即报错。

## 复用声明

本仓库为独立编写，未复制课程标准仓库或网上项目的代码。方法（分组切分、质量门、模型包核对、
可加性解释、规格卡驱动的开发顺序）参考课程讲义与任务书；实现、契约、门槛与领域代码均为本项目原创。