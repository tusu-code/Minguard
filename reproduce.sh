#!/usr/bin/env bash
# 从打分文件重新生成论文的全部表格（tables/）
set -euo pipefail
cd "$(dirname "$0")/src"
test -f ../data/dev_20240627/dev.json || { echo "请先把 BIRD dev（dev_20240627）解压到 data/dev_20240627/"; exit 1; }
export FACTS_VERSION=1
python label_budget.py
python pg_summary.py 2>/dev/null || echo "（runs/pg_bench.json 缺失时跳过 PostgreSQL 汇总）"
python make_tables.py
python make_figures.py --recompute
