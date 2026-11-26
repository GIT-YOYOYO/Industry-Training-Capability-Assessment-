# 规格卡 003：模型包与推理

## 目标

交付一个别人拿去就能用、用错了会被拦住的模型包，以及给出预测+决策+解释的推理入口。

## 输入

- `models/champion/`（由 train 生成）

## 输出

- `models/champion/model.joblib` + `metadata.json`
- `predict` 命令的 JSON 输出（每行：概率、决策、基准值、贡献、Top 因素）

## 验收标准

1. 模型包元数据含：模型名、特征清单、决策规则、贡献单位、配置/数据指纹、标签阈值、依赖版本。
2. `predict --input` 输出概率、决策、解释；`base + Σ贡献 = logit`。
3. 输入缺列/缺失 -> 清楚报错，不静默填充。
4. 加载时特征清单或依赖主版本不符 -> 拒绝加载并说明原因。

## 不做

- 不做 HTTP 服务（X8 未认领）。
- 不做前端页面。

## 验证命令

```bash
uv run vtrain predict --input samples.csv
uv run vtrain explain --input samples.csv --row 0
```