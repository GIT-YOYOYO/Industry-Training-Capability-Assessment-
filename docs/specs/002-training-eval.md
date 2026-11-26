# 规格卡 002：训练与评估

## 目标

训练基线+候选，验证集选 champion、定决策规则，测试集只评估一次，生成由代码驱动的报告。

## 输入

- `data/processed/*.parquet`、`manifest.json`
- `configs/base.yaml`（超参、门槛、种子）

## 输出

- `reports/evaluation.{json,md}`
- `runs/<config指纹>/run.json`（追踪）

## 验收标准

1. 报告含：模型对比（验证集）、champion 测试集指标、自助 95% CI、切片（PR-AUC/ROC-AUC/lift）、
   校准（Brier/ECE）、错误分析、全局解释、门禁结论。
2. 测试集只在最终评估用一次；champion 由验证集 PR-AUC 选出。
3. 训练门禁三条：主指标比基线高 >=0.05、切片 ROC-AUC 差距 <=0.20、解释可加回（<=1e-4）。
4. 删掉报告重跑，内容一致（固定种子；同一机器）。

## 不做

- 不做服务、不做回退/注册表。
- 不手改报告数字。

## 验证命令

```bash
uv run vtrain train
sed -n '1,40p' reports/evaluation.md
```