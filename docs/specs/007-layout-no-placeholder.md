# 规格卡 007：布局与无占位守门

## 目标

把“仓库长什么样”和“不许留占位”也用测试钉住（X16 的第二类）：目录布局、关键文件、类型标记、
以及源码里不得出现 TODO/FIXME/NotImplementedError 和裸 `# type: ignore`。

## 输入

- 仓库根、`src/vtrain/`、`tests/`

## 输出

- `src/vtrain/py.typed`（类型包标记）
- `tests/architecture/test_layout.py`

## 验收标准

1. 关键文件存在：README、AGENTS、pyproject、uv.lock、configs/*、`.importlinter`、`scripts/ci.py`。
2. 测试四类目录存在：unit / acceptance / e2e / architecture。
3. `src/vtrain/py.typed` 存在。
4. `src/` 下没有 TODO / FIXME / XXX / NotImplementedError。
5. `src/` 下没有不带错误码的裸 `# type: ignore`（必须写成 `# type: ignore[code]`）。

## 不做

- 不把测试目录重排。
- 不引入新的占位文件。