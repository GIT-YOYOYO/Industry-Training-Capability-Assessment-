# ADR-0007：自建轻量追踪协议，不引入 MLflow 外部依赖

## 背景

X7 要求记录参数、指标、工件，外加 git sha、是否未提交、配置指纹、数据指纹；测试要用内存后端、不连真服务。

## 决定

定义一个 `Tracker` 协议（set_context / log_params / log_metrics / log_artifact / end），
实现 `JsonlTracker`（写 run.json）与 `MemoryTracker`（测试用）。不引入 MLflow/W&B。

## 后果与代价

- 零外部依赖、离线可复现、测试不连服务；换后端只需实现同一协议，调用方不改。
- 代价：没有 MLflow 的 UI、对比视图和模型注册功能；需要时再按协议接一个适配器即可。