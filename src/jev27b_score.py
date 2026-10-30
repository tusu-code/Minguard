"""把 GPU 机上 JEV-27B 的逐问题概率融合成守卫分数（与托管 Jev 相同的 guards.fuse），写 runs/scores_jev27b.jsonl。
确定性层直接判定的请求沿用主系统的结果（两者事实层相同）。用法：python jev27b_score.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import guards as G
import paper_data as P


def main() -> None:
    probs: dict = defaultdict(dict)
    ind = {}
    for l in open(P.RUNS / "jev27b_probs.jsonl"):
        r = json.loads(l)
        probs[(r["guard"], r["id"])][r["q"]] = r["p"]
        ind[r["id"]] = r["individual"]
    with open(P.RUNS / "scores_jev27b.jsonl", "w") as fo:
        for (g, i), ps in probs.items():
            if g == "jev27b_decomposed":
                s, why = G.fuse(ps, individual=ind[i])
            else:
                s, why = ps["violation"], "violation"
            fo.write(json.dumps({"guard": g, "id": i, "score": s, "why": why, "probs": ps}) + "\n")
    print(len(probs), "个守卫分数")


if __name__ == "__main__":
    main()
