# 规格卡 005：分层契约（import-linter）

## 目标

用机器强制 `vtrain` 内部的依赖方向，防止低层模块反向依赖高层，把“架构约束”从口头约定变成质量门里会红的一道。

## 分层

- 基础层：`vtrain.config`、`vtrain.tracking`（不依赖其它 vtrain 模块）
- 中层：`vtrain.data`、`vtrain.features`、`vtrain.evaluation`、`vtrain.explain`、`vtrain.bundle`
- 高层：`vtrain.models`、`vtrain.predict`、`vtrain.cli`

## 输入

- `src/vtrain/` 的 import 关系

## 输出

- 仓库根新增 `.importlinter`（声明 forbidden 契约）
- `scripts/ci.py` 增加一道 `import-linter` 门

## 验收标准

1. 根目录存在 `.importlinter`，且至少声明一条 `forbidden` 契约。
2. `scripts/ci.py` 会调用 `lint-imports`（作为质量门的一道）。
3. `lint-imports` 在现有代码上通过（无违规）。

## 不做

- 不把单包重构成多包工作区。
- 不改任何业务代码的 import（除非确有违规）。