# 规格卡 006：质量门元测试（覆盖率门槛单一来源）

## 目标

让“质量门本身”也被测试守住（X16）：门的数量与顺序固定，覆盖率门槛只有一个出处（pyproject），
`ci.py` 里不允许再写死一个数字。

## 输入

- `scripts/ci.py`、`pyproject.toml`

## 输出

- `pyproject.toml` 增加 `[tool.coverage.run]` 与 `[tool.coverage.report] fail_under`
- `ci.py` 从 pyproject 读覆盖率门槛（删除 `COVERAGE_MIN` 常量）

## 验收标准

1. 质量门恰好 5 道，顺序为 ruff lint → ruff format → mypy → import-linter → pytest。
2. pytest 门的 `--cov-fail-under` 值等于 `pyproject.toml` 的 `tool.coverage.report.fail_under`。
3. `ci.py` 源码中不含 `COVERAGE_MIN`（不允许写死门槛）。

## 不做

- 不改测试内容或用例数。
- 不降低覆盖率门槛（保持 >= 80，本项目取 85）。