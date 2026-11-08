"""把 GPU 机上 27B 底座读出的逐问题概率融合成守卫分数（guards.fuse，与主系统相同），写 runs/scores_base27b.jsonl。
用法：python base27b_score.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import guards as G
import paper_data as P


def main() -> None:
    probs: dict = defaultdict(dict)
    ind = {}
    for l in open(P.RUNS / "base27b_probs.jsonl"):
        r = json.loads(l)
        probs[(r["guard"], r["id"])][r["q"]] = r["p"]
        ind[r["id"]] = r["individual"]
    with open(P.RUNS / "scores_base27b.jsonl", "w") as fo:
        for (g, i), ps in probs.items():
            if g == "base27b_decomposed":
                s, why = G.fuse(ps, individual=ind[i])
            else:
                s, why = ps["violation"], "violation"
            fo.write(json.dumps({"guard": g, "id": i, "score": s, "why": why, "probs": ps}) + "\n")
    print(len(probs), "guard scores")


if __name__ == "__main__":
    main()
