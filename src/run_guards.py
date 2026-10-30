"""对基准逐条跑全部守卫，结果写 runs/scores.jsonl（每行一条样本 × 一个守卫）。

用法：python3 run_guards.py [--limit N] [--guards a,b,c] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import dbmeta
import guards as G
import sqlfacts
from jev_client import JevClient
from llm_client import LLMClient

BENCH = dbmeta.ROOT / "data/pilot_bench.jsonl"
OUT = dbmeta.ROOT / "runs/scores.jsonl"

INNER = None   # 可缓存问法一条样本要发多个请求，用一个共享线程池并行发

ALL = ["rule_personal", "rule_acl", "jev_holistic_raw", "jev_holistic", "jev_decomposed",
       "llm_deepseek", "llm_qwen235b"]
LLM_MODELS = {"llm_deepseek": "deepseek/deepseek-v3.2", "llm_qwen235b": "qwen/qwen3-235b-a22b-2507"}


def run_one(name: str, item: dict, facts: sqlfacts.Facts, clients: dict) -> dict:
    t0 = time.monotonic()
    sc = G.short_circuit(item, facts)
    if sc is not None and name != "jev_holistic_raw":
        s, why = sc
        res = {"score": s, "why": why, "det": True}
    elif name == "rule_personal":
        res = G.rule_block_personal(item, facts)
    elif name == "rule_acl":
        res = G.rule_compiled_acl(item, facts)
    elif name == "jev_holistic_raw":
        # 不给确定性事实、不短路：只看问题、策略、SQL
        res = G.jev_holistic(clients["jev"], item, facts, with_facts=False)
    elif name == "jev_holistic":
        res = G.jev_holistic(clients["jev"], item, facts, with_facts=True)
    elif name == "jev_decomposed":
        res = G.jev_decomposed(clients["jev"], item, facts)
    elif name == "jev_cacheable":
        res = G.jev_cacheable(clients["jev"], item, facts, pool=INNER)
    elif name in LLM_MODELS:
        res = G.llm_holistic(clients[name], item, facts)
    else:
        raise ValueError(name)
    res.update({"guard": name, "id": item["id"], "latency_ms": (time.monotonic() - t0) * 1000})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--guards", default=",".join(ALL))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--bench", default=str(BENCH))
    args = ap.parse_args()
    names = args.guards.split(",")
    items = [json.loads(l) for l in open(args.bench)]
    if args.limit:
        step = max(1, len(items) // args.limit)
        items = items[::step][: args.limit]

    global INNER
    INNER = ThreadPoolExecutor(32)
    os.environ.setdefault("JEV_CACHE", str(dbmeta.ROOT / "runs/jev_cache.sqlite"))
    clients = {"jev": JevClient(backend="openrouter", max_rps=15)}
    for n in names:
        if n in LLM_MODELS:
            clients[n] = LLMClient(LLM_MODELS[n])

    t0 = time.monotonic()
    facts = {it["id"]: sqlfacts.facts_for(it["db"], it["sql"]) for it in items}
    print(f"确定性层 {len(items)} 条，用时 {time.monotonic() - t0:.1f}s", flush=True)

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            done.add((r["guard"], r["id"]))
    jobs = [(n, it) for n in names for it in items if (n, it["id"]) not in done]
    print(f"待跑 {len(jobs)} 个（已有 {len(done)}）", flush=True)
    errors = 0
    with open(args.out, "a") as fo, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_one, n, it, facts[it["id"]], clients): (n, it["id"]) for n, it in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                fo.write(json.dumps(fu.result(), ensure_ascii=False) + "\n")
                fo.flush()
            except Exception as e:  # noqa: BLE001
                errors += 1
                print("失败", futs[fu], str(e)[:200], file=sys.stderr, flush=True)
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    j = clients["jev"].stats
    print(f"Jev：请求 {j.requests}，缓存命中 {j.cache_hits}，输入 {j.input_tokens} token，"
          f"约 ${j.cost_usd():.4f}，模型 {j.by_model}")
    for n in names:
        if n in LLM_MODELS:
            s = clients[n].stats
            print(f"{n}：请求 {s.requests}，缓存 {s.cache_hits}，token {s.prompt_tokens}+{s.completion_tokens}，"
                  f"${s.cost:.4f}")
    print(f"失败 {errors}")


if __name__ == "__main__":
    main()
