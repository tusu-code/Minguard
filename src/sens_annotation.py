"""敏感性实验：个人列标注不完整时，fact layer 会直接放行多少正例。

把每个库的个人列标注展开成逐列清单（通配表展开成全部列），每次随机删掉比例 p 的列，
用同一批事实（解析与试执行只做一次）重新走短路规则，统计原本送进判官、现在被放行的正例。
用法：FACTS_VERSION=1 python sens_annotation.py   → runs/sens_annotation.json
"""

from __future__ import annotations

import copy
import json
import random

import dbmeta
import guards as G
import paper_data as P
import sqlfacts

FRACS = [0.05, 0.10, 0.20, 0.30]
REPS = 50


def explicit_personal() -> dict:
    out = {}
    for db, tabs in dbmeta.PERSONAL.items():
        out[db] = {}
        for t, cols in tabs.items():
            tt, _ = dbmeta._canon(db, t)
            if tt is None:
                continue
            full = {}
            for c in dbmeta.load_schema(db)[tt]:
                cat = cols.get(c) or cols.get(dbmeta._ALL)
                if cat:
                    full[c] = cat
            out[db][t] = full
    return out


def main() -> None:
    full = explicit_personal()
    cols = [(db, t, c) for db in full for t in full[db] for c in full[db][t]]
    pos = {}
    human = {r["id"] for r in json.load(open(dbmeta.ROOT / "annotation/human_labels.json")) if r["verdict"] != "不越量"}
    for b in ["heldout", "real", "new_model", "new_db", "real_full"]:
        it, S = P.load(b)
        for i in P.judged(it, S):
            if it[i]["label"] or (b == "real_full" and i in human):
                pos.setdefault(i, (it[i], set()))[1].add(b if b != "real_full" or it[i]["label"] else "human")
                if b == "real_full" and i in human:
                    pos[i][1].add("human")
    for x in (json.loads(l) for l in open(P.DATA / "human_extra.jsonl")):
        if x["id"] in human:
            pos.setdefault(x["id"], (x, set()))[1].add("human")
    facts = {i: sqlfacts.facts_for(x["db"], x["sql"]) for i, (x, _) in pos.items()}
    groups = ["heldout", "real", "new_model", "new_db", "human"]

    def allowed() -> set:
        out = set()
        for i, (x, _) in pos.items():
            r = G.short_circuit(x, facts[i])
            if r is not None and r[0] == 0.0:
                out.add(i)
        return out

    orig = dbmeta.PERSONAL
    dbmeta.PERSONAL = copy.deepcopy(full)
    base = allowed()
    res = {"n_cols": len(cols), "n_pos": {g: sum(g in s for _, s in pos.values()) for g in groups},
           "base_allowed": len(base), "fracs": {}}
    rng = random.Random(0)
    for p in FRACS:
        per = {g: [] for g in groups}
        for _ in range(REPS):
            drop = set(rng.sample(cols, round(p * len(cols))))
            P2 = copy.deepcopy(full)
            for db, t, c in drop:
                del P2[db][t][c]
            dbmeta.PERSONAL = P2
            a = allowed()
            for g in groups:
                n = res["n_pos"][g]
                per[g].append(sum(1 for i in a if g in pos[i][1]) / n if n else 0.0)
        res["fracs"][str(p)] = {g: {"mean": sum(v) / len(v), "max": max(v)} for g, v in per.items()}
        print(p, {g: round(res["fracs"][str(p)][g]["mean"], 3) for g in groups})
    dbmeta.PERSONAL = orig
    json.dump(res, open(P.RUNS / "sens_annotation.json", "w"), indent=1)
    print(res["n_cols"], res["n_pos"], res["base_allowed"])


if __name__ == "__main__":
    main()
