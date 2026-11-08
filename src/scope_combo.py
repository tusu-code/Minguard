"""scope 组合检验（离线）：MinGuard 的逐问题概率里，scope 一问换成 DeepSeek-V3.2 的回答或取两者最大，重新融合。
不调用任何模型：DeepSeek 的 scope 概率取自 runs/scores_llm_decomp.jsonl。
用法：FACTS_VERSION=1 python scope_combo.py
"""

from __future__ import annotations

import json

import evaluate as E
import guards as G
import make_tables as T
import paper_data as P

BENCHES = {"heldout": "Held-out", "real": "Real", "new_model": "New model", "new_db": "New DBs"}


def variants(S: dict, ind: dict) -> None:
    ours, ds = S["jev_decomposed"], S["llm_deepseek_decomposed"]
    for name in ("scope_ds", "scope_max"):
        S[name] = {}
    miss = 0
    for i, r in ours.items():
        if r.get("det") or not r.get("probs"):
            S["scope_ds"][i] = S["scope_max"][i] = r
            continue
        p = dict(r["probs"])
        q = ds.get(i, {}).get("probs", {})
        if "scope" not in q:
            miss += 1
            S["scope_ds"][i] = S["scope_max"][i] = r
            continue
        a, b = dict(p), dict(p)
        a["scope"] = q["scope"]
        b["scope"] = max(p.get("scope", 0.0), q["scope"])
        S["scope_ds"][i] = {"score": G.fuse(a, ind[i])[0]}
        S["scope_max"][i] = {"score": G.fuse(b, ind[i])[0]}
    return miss


def main() -> None:
    ind = json.load(open(P.RUNS / "individual.json"))
    out = {}
    for b, name in BENCHES.items():
        it, S = P.load(b)
        miss = variants(S, ind)
        J = P.judged(it, S)
        te = P.test_half(it, J)
        row = {"missing_ds_scope": miss}
        for g in ("jev_decomposed", "scope_ds", "scope_max", "llm_deepseek_decomposed", "llm_qwen235b"):
            row[g] = {"auc": T.auc(it, S, g, te)}
            if g != "jev_decomposed":
                m, lo, hi, _ = E.boot_auc_diff_grouped(it, te, S["jev_decomposed"], S[g])
                row[g]["minguard_minus"] = [m, lo, hi]
            r = E.grouped_risk(it, S, g, J, 0.05, 0.10, reps=200)
            row[g]["risk"] = {k: r[k][0] for k in ("auto", "miss", "false_block")}
        # extra-people 正例（strict，测试半边）的检出：分数 ≥ 0.5
        ep = [i for i in te if it[i]["label"] == 1 and it[i]["category"] == "real:extra_people"]
        neg = [i for i in te if it[i]["label"] == 0]
        row["extra_people"] = {g: [sum(S[g][i]["score"] >= 0.5 for i in ep), len(ep)]
                               for g in ("jev_decomposed", "scope_ds", "scope_max", "llm_qwen235b")}
        row["neg_flag_rate"] = {g: sum(S[g][i]["score"] >= 0.5 for i in neg) / len(neg)
                                for g in ("jev_decomposed", "scope_ds", "scope_max", "llm_qwen235b")}
        out[b] = row
        print(name, json.dumps(row, indent=1))
    json.dump(out, open(P.RUNS / "scope_combo.json", "w"), indent=1)


if __name__ == "__main__":
    main()
