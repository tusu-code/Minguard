"""确定性层各步耗时：解析+血缘、试执行（封顶取行）、聚合人数改写查询，以及短路比例。

用法：python bench_latency.py ../data/real_bench_strict.jsonl ../data/final_bench.jsonl
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import guards as G
import sqlfacts


def main() -> None:
    items = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
    t_parse, t_exec, t_contrib, total = [], [], [], []
    short = {"block": 0, "allow": 0, "judge": 0}
    for it in items:
        db, sql = it["db"], it["sql"]
        t0 = time.perf_counter()
        f = sqlfacts.analyze(db, sql)
        f.conditions = sqlfacts.top_conditions(sql)
        t1 = time.perf_counter()
        if f.statement == "select" and not f.parse_error:
            sqlfacts.dry_run(db, sql, f)
        t2 = time.perf_counter()
        if f.aggregate_query and not f.exec_error:
            f.contributors = sqlfacts.contributors(db, sql)
        t3 = time.perf_counter()
        t_parse.append((t1 - t0) * 1e3); t_exec.append((t2 - t1) * 1e3); t_contrib.append((t3 - t2) * 1e3)
        total.append((t3 - t0) * 1e3)
        sc = G.short_circuit(it, f)
        short["judge" if sc is None else ("block" if sc[0] >= 0.5 else "allow")] += 1

    def q(v):
        v = np.asarray(v)
        return f"p50 {np.percentile(v, 50):7.2f}  p95 {np.percentile(v, 95):8.2f}  p99 {np.percentile(v, 99):8.2f}"
    print(f"{len(items)} 条")
    print("  解析+血缘   ", q(t_parse))
    print("  试执行      ", q(t_exec))
    print("  聚合人数    ", q(t_contrib))
    print("  合计        ", q(total))
    n = len(items)
    print("  短路：", {k: f"{v} ({v / n:.1%})" for k, v in short.items()})
    if os.environ.get("LAT_OUT"):
        def pct(v):
            return {p: float(np.percentile(v, p)) for p in (50, 95, 99)}
        json.dump({"n": n, "parse": pct(t_parse), "exec": pct(t_exec), "contrib": pct(t_contrib),
                   "total": pct(total), "short": short}, open(os.environ["LAT_OUT"], "w"), indent=1)


if __name__ == "__main__":
    main()
