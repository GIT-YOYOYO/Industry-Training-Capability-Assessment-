# 规格卡 004：质量门

## 目标

一条命令决定合不合格，并在出错时真的变红。

## 输入

- `src/`、`tests/`、`scripts/`

## 输出

- 终端结论：全绿或 `[FAIL]`

## 验收标准

1. 顺序执行 ruff lint、ruff format --check、mypy strict、pytest，任一失败即返回非 0。
2. 覆盖率门槛（>=80%）只写在 `scripts/ci.py` 一处。
3. 破坏实验：改坏一处关键代码或调低门槛，质量门变红（见 `docs/self-assessment.md`）。

## 不做

- 不接远程 CI（X18 未认领）。
- 不做变异测试（X17 未认领）。

## 验证命令

```bash
uv run python scripts/ci.py
```