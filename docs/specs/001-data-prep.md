# 规格卡 001：数据准备

## 目标

从原始 xlsx 得到按中心分组切分的 train/val/test 与一份可核对的 manifest。

## 输入

- `data/raw/kaggledata225.xlsx`（sha256 与 `configs/base.yaml` 一致）
- `configs/base.yaml`、`configs/features.yaml`

## 输出

- `data/processed/{train,val,test}.parquet`
- `data/processed/manifest.json`

## 验收标准

1. 清洗：`y/y1..y5` 缺失行被剔除，`manifest.cleaning` 记录删了多少、缺了哪些列。
2. 切分：三个切分的 `schoolid` 互不相交（`group_disjoint=true`）。
3. 标签：阈值只在训练集上算（`label_threshold`），val/test 复用同一阈值。
4. manifest 含各切分行数、中心数、正例率、内容指纹、配置指纹、原始 sha256。
5. 特征列无缺失；入模列只来自 `features.yaml`。

## 不做

- 不做特征工程（下一步）。
- 不把原始数据写进 Git。

## 验证命令

```bash
uv run vtrain prepare
cat data/processed/manifest.json
```