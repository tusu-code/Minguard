"""标注量曲线：风险控制需要多少个带标签的目标题目。

与 evaluate.grouped_risk 相同的按题分组 conformal（校准侧每题随机取 1 条，测试侧按 1/题内条数加权），
只是校准题目数固定为 n，其余题目全部作测试。每个 n 做 500 次随机划分，报告误放、误拦的均值、
单次划分超出预算的比例，以及自动化率。题目不足 n + 20 个的工作负载跳过该 n。
用法：FACTS_VERSION=1 python label_budget.py   → runs/label_budget.json
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import evaluate as E
import paper_data as P

NS = [25, 50, 100, 200]
REPS = 500
ALPHA, BETA = 0.05, 0.10


def curve(items, S, guard, ids, n, reps=REPS, seed=0):
    rng = np.random.default_rng(seed)
    by_q = defaultdict(list)
    for i in ids:
        by_q[(items[i]["db"], items[i]["qid"], items[i]["policy"])].append(i)
    qs = list(by_q)
    if len(qs) < n + 20:
        return None
    out = defaultdict(list)
    for _ in range(reps):
        perm = rng.permutation(len(qs))
        cal_q, te_q = [qs[k] for k in perm[:n]], [qs[k] for k in perm[n:]]
        cal = [by_q[q][rng.integers(len(by_q[q]))] for q in cal_q]
        y = [items[i]["label"] for i in cal]
        l, h = E.thresholds(y, [S[guard][i]["score"] for i in cal], ALPHA, BETA)
        wpos = wneg = mp = fb = auto = allow = wall = 0.0
        for q in te_q:
            w = 1.0 / len(by_q[q])
            for i in by_q[q]:
                s, lab = S[guard][i]["score"], items[i]["label"]
                wall += w
                auto += w * (s < l or s >= h)
                allow += w * (s < l)
                if lab:
                    wpos += w; mp += w * (s < l)
                else:
                    wneg += w; fb += w * (s >= h)
        miss, fbr = mp / max(wpos, 1e-9), fb / max(wneg, 1e-9)
        out["miss"].append(miss); out["false_block"].append(fbr)
        out["auto"].append(auto / wall); out["allow"].append(allow / wall)
        out["over_miss"].append(miss > ALPHA); out["over_fb"].append(fbr > BETA)
        out["n_pos_cal"].append(sum(y))
    return {k: float(np.mean(v)) for k, v in out.items()} | {"n_questions": len(qs)}


def main() -> None:
    res = {}
    for b in ["heldout", "real", "new_model", "new_db"]:
        it, S = P.load(b)
        J = P.judged(it, S)
        for g in ["jev_decomposed", "jev27b_decomposed"]:
            for n in NS:
                r = curve(it, S, g, J, n)
                if r is None:
                    continue
                res[f"{b}|{g}|{n}"] = r
                print(f"{b:9s} {g:18s} n={n:3d} pos_cal={r['n_pos_cal']:5.1f} miss={r['miss']:.3f} "
                      f"P(miss>a)={r['over_miss']:.2f} fb={r['false_block']:.3f} P(fb>b)={r['over_fb']:.2f} "
                      f"auto={r['auto']:.3f} allow={r['allow']:.3f}", flush=True)
    json.dump(res, open(P.RUNS / "label_budget.json", "w"), indent=1)


if __name__ == "__main__":
    main()
