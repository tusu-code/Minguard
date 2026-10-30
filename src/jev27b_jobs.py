"""开源复现 JEV-27B 重跑 MinGuard：本地生成全部决策请求（需要 BIRD 库算事实），GPU 机只做推理。

state 与托管 Jev 完全相同（guards.build_state）；问题文本也相同，按 JEV-27B 的 bare-v1 模板写成
"[kind] noul / [state] / [question] / [options] false true"。noul 的问题按模型说明的建议改成问句，
让 "true" 对应原 criteria 的 "Yes, this statement is true for the given request."。
输出 runs/jev27b_jobs.jsonl：每行一个 (样本, 守卫, 问题名) 的 prompt；融合在 jev27b_score.py 里做。
用法：FACTS_VERSION=1 python jev27b_jobs.py
"""

from __future__ import annotations

import json

import guards as G
import paper_data as P
import sqlfacts

OUT = P.RUNS / "jev27b_jobs.jsonl"


def question(text: str) -> str:
    return f"Is the following statement true for this request? {text}"


def prompt(state: dict, q: str) -> str:
    return (f"[kind] noul\n[state] {json.dumps(state, ensure_ascii=False)}\n[question] {question(q)}\n"
            f"[options]\nfalse\ntrue\n[decision]:")


def items_to_judge() -> dict:
    out = {}
    for b in ["heldout", "real_full", "new_model", "new_db"]:
        it, S = P.load(b)
        for i in P.judged(it, S):
            out[i] = it[i]
    det = {}
    for l in open(P.RUNS / "scores_human_extra.jsonl"):
        r = json.loads(l)
        if r["guard"] == "jev_decomposed":
            det[r["id"]] = r
    for l in open(P.DATA / "human_extra.jsonl"):
        x = json.loads(l)
        if x["id"] not in out and not det[x["id"]].get("det"):
            out[x["id"]] = x
    return out


def main() -> None:
    items = items_to_judge()
    n = 0
    with open(OUT, "w") as fo:
        for i, x in items.items():
            f = sqlfacts.facts_for(x["db"], x["sql"])
            ind = bool(f.individual_level(x["db"]))
            st = G.build_state(x, f, with_facts=True)
            for k, q in G.decomposed_questions(x, f).items():
                fo.write(json.dumps({"id": i, "guard": "jev27b_decomposed", "q": k, "individual": ind,
                                     "prompt": prompt(st, q["instructions"])}, ensure_ascii=False) + "\n")
                n += 1
            fo.write(json.dumps({"id": i, "guard": "jev27b_holistic", "q": "violation", "individual": ind,
                                 "prompt": prompt(st, G.HOLISTIC)}, ensure_ascii=False) + "\n")
            n += 1
    print(len(items), "条请求，", n, "个 prompt")


if __name__ == "__main__":
    main()
