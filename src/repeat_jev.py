"""Jev 重复调用方差：从两个最终工作负载的判官请求里分层抽 100 条，不走缓存各调用 3 次。

报告每条请求融合分数的标准差、决策（固定阈值下 allow/escalate/block）改变的比例、每次运行的 AUROC。
用法：FACTS_VERSION=1 python repeat_jev.py   → runs/repeat_jev.jsonl
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor

import guards as G
import paper_data as P
import sqlfacts
from jev_client import JevClient

N, REPS = 100, 3
OUT = P.RUNS / "repeat_jev.jsonl"


def sample() -> list[dict]:
    rng = random.Random(0)
    out = []
    for b in ["new_model", "new_db"]:
        it, S = P.load(b)
        J = P.judged(it, S)
        for y in (1, 0):
            ids = sorted(i for i in J if it[i]["label"] == y)
            out += [it[i] for i in rng.sample(ids, N // 4)]
    return out


def main() -> None:
    items = sample()
    jev = JevClient(backend="openrouter", max_rps=15)
    facts = {x["id"]: sqlfacts.facts_for(x["db"], x["sql"]) for x in items}

    def one(args):
        rep, x = args
        f = facts[x["id"]]
        r = jev.ask(G.build_state(x, f), G.decomposed_questions(x, f), use_cache=False)
        probs = {k: float(v["noul"]) for k, v in r["answers"].items()}
        s, why = G.fuse(probs, individual=bool(f.individual_level(x["db"])))
        return {"rep": rep, "id": x["id"], "block": x["block"], "label": x["label"], "score": s, "why": why,
                "probs": probs}

    with open(OUT, "w") as fo, ThreadPoolExecutor(10) as ex:
        for r in ex.map(one, [(k, x) for k in range(REPS) for x in items]):
            fo.write(json.dumps(r) + "\n")
    print(f"请求 {jev.stats.requests}，约 ${jev.stats.cost_usd():.4f}")


if __name__ == "__main__":
    main()
