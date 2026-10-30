"""融合方式消融：同一组逐问题概率，换融合规则重算分数。

违规项与 guards.fuse 相同：need_i 取 1-p，forbid_i、scope 直接取 p，aggregate_only 只在输出为个体级时计入。
  max      主系统（任一项违规即违规）
  mean     各违规项平均
  noisyor  1 - Π(1 - v)
  learned  逻辑回归加权，特征为每类违规项的最大值（need/forbid/aggregate/scope），只在开发集（合成试点）上训练，
           与"只用开发集训练的分类器"一样不看目标标签
输出 runs/scores_fusion.jsonl（守卫名 fuse_<规则>）；"个体级"标记缓存在 runs/individual.json。
用法：FACTS_VERSION=1 python fusion_ablation.py
"""

from __future__ import annotations

import json
import math

import numpy as np
from sklearn.linear_model import LogisticRegression

import paper_data as P
import sqlfacts

KINDS = ["need", "forbid", "aggregate_only", "scope"]


def terms(probs: dict, individual: bool) -> dict:
    t = {k: [] for k in KINDS}
    for k, p in probs.items():
        if k.startswith("need_"):
            t["need"].append(1.0 - p)
        elif k.startswith("forbid_"):
            t["forbid"].append(p)
        elif k == "scope":
            t["scope"].append(p)
        elif k == "aggregate_only" and individual:
            t["aggregate_only"].append(p)
    return t


def feats(t: dict) -> list[float]:
    return [max(t[k]) if t[k] else 0.0 for k in KINDS]


def main() -> None:
    cache_p = P.RUNS / "individual.json"
    ind = json.load(open(cache_p)) if cache_p.exists() else {}
    data = {}
    for b in ["pilot", "heldout", "real_full", "new_model", "new_db"]:
        it, S = P.load(b)
        for i, r in S["jev_decomposed"].items():
            if r.get("det") or not r.get("probs"):
                continue
            if i not in ind:
                x = it[i]
                ind[i] = bool(sqlfacts.facts_for(x["db"], x["sql"], execute=False).individual_level(x["db"]))
            data[i] = (r["probs"], ind[i], it[i]["label"], b)
    # 200 条人工样本里有 49 条不在评测子集，打分在 scores_human_extra.jsonl
    extra = {json.loads(l)["id"]: json.loads(l) for l in open(P.DATA / "human_extra.jsonl")}
    for l in open(P.RUNS / "scores_human_extra.jsonl"):
        r = json.loads(l)
        if r["guard"] != "jev_decomposed" or r.get("det") or not r.get("probs") or r["id"] in data:
            continue
        x = extra[r["id"]]
        if r["id"] not in ind:
            ind[r["id"]] = bool(sqlfacts.facts_for(x["db"], x["sql"], execute=False).individual_level(x["db"]))
        data[r["id"]] = (r["probs"], ind[r["id"]], x.get("label", 0), "human_extra")
    json.dump(ind, open(cache_p, "w"))
    X = [feats(terms(p, f)) for i, (p, f, y, b) in data.items() if b == "pilot"]
    Y = [y for i, (p, f, y, b) in data.items() if b == "pilot"]
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(X, Y)
    print("learned weights", dict(zip(KINDS, np.round(lr.coef_[0], 2))), "bias", round(float(lr.intercept_[0]), 2),
          "train n", len(Y))
    with open(P.RUNS / "scores_fusion.jsonl", "w") as fo:
        for i, (p, f, y, b) in data.items():
            t = terms(p, f)
            v = [x for k in KINDS for x in t[k]]
            sc = {"max": max(v) if v else 0.0,
                  "mean": float(np.mean(v)) if v else 0.0,
                  "noisyor": 1.0 - math.prod(1.0 - x for x in v) if v else 0.0,
                  "learned": float(lr.predict_proba([feats(t)])[0, 1])}
            for k, s in sc.items():
                fo.write(json.dumps({"guard": f"fuse_{k}", "id": i, "score": s}) + "\n")
    print(len(data), "requests")


if __name__ == "__main__":
    main()
