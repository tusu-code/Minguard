"""守卫端到端压测：确定性层 + Jev 分解问法，不用缓存（每次都真实调用），不同并发度下的吞吐与延迟。

工作负载：OmniSQL-7B 在原 4 库上的自然输出分布（default 策略，去重后随机抽 N 条）。
用法：python bench_throughput.py --n 400 --concurrency 1,8,32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import dbmeta
import guards as G
import sqlfacts
from jev_client import JevClient


def guard_one(jev, it: dict) -> dict:
    t0 = time.perf_counter()
    f = sqlfacts.facts_for(it["db"], it["sql"])
    t1 = time.perf_counter()
    sc = G.short_circuit(it, f)
    if sc is not None:
        return {"det_ms": (t1 - t0) * 1e3, "judge_ms": 0.0, "total_ms": (t1 - t0) * 1e3, "judged": False}
    r = G.jev_decomposed(jev, it, f)
    t2 = time.perf_counter()
    return {"det_ms": (t1 - t0) * 1e3, "judge_ms": (t2 - t1) * 1e3, "total_ms": (t2 - t0) * 1e3,
            "judged": True, "tokens": r["tokens"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--concurrency", default="1,8,32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    xs = [json.loads(l) for l in open(dbmeta.ROOT / "data/real_bench_raw.jsonl")]
    xs = [x for x in xs if x["policy"] == "default" and not x["category"].startswith("real:non-select")]
    work = random.Random(args.seed).sample(xs, args.n)
    out = {}
    for c in map(int, args.concurrency.split(",")):
        cache = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name
        jev = JevClient(backend="openrouter", cache_path=cache, max_rps=20)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(lambda it: guard_one(jev, it), work))
        wall = time.perf_counter() - t0
        tot = np.array([r["total_ms"] for r in res])
        jud = np.array([r["judge_ms"] for r in res if r["judged"]])
        row = {"concurrency": c, "n": len(res), "wall_s": wall, "qps": len(res) / wall,
               "judged_frac": float(np.mean([r["judged"] for r in res])),
               "total_p50": float(np.percentile(tot, 50)), "total_p95": float(np.percentile(tot, 95)),
               "total_p99": float(np.percentile(tot, 99)),
               "judge_p50": float(np.percentile(jud, 50)), "judge_p95": float(np.percentile(jud, 95)),
               "jev_requests": jev.stats.requests, "jev_retries": jev.stats.retries,
               "cost_usd": jev.stats.cost_usd()}
        out[c] = row
        print(f"并发 {c:3d}: {row['qps']:6.2f} 查询/秒，端到端 p50 {row['total_p50']:6.0f} ms p95 "
              f"{row['total_p95']:6.0f} ms p99 {row['total_p99']:6.0f} ms；判官 p50 {row['judge_p50']:6.0f} ms；"
              f"进判官 {row['judged_frac']:.1%}，重试 {row['jev_retries']}，${row['cost_usd']:.4f}", flush=True)
        jev.close()
        os.unlink(cache)
    json.dump(out, open(dbmeta.ROOT / "runs/throughput.json", "w"), indent=1)


if __name__ == "__main__":
    main()
