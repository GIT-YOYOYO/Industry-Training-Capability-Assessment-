# 架构决定记录（ADR）

每条都很短：背景、决定、后果与代价。

| 编号 | 决定 |
|---|---|
| [0001](0001-task-definition.md) | 任务定义为“不达标风险”二分类，主指标 PR-AUC |
| [0002](0002-schooltype-excluded.md) | 机构类型 schooltype 不入模（公平性） |
| [0003](0003-composite-x-excluded.md) | 综合分 x 不入模，只用 x1–x5（冗余与共线性） |
| [0004](0004-group-split.md) | 按培训中心分组切分，中心互不相交 |
| [0005](0005-slice-gate-roc-auc.md) | 切片门禁改用 ROC-AUC（PR-AUC 受正例率影响） |
| [0006](0006-topk-batch-decision.md) | 名额制按批内 Top-K，单条用验证集等效阈值 |
| [0007](0007-lightweight-tracking.md) | 自建轻量追踪协议，不引入 MLflow 外部依赖 |