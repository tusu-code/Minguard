"""审稿意见补的独立基线：JEV-27B 的底座（Qwen 27B，不加载 Jev 蒸馏出的 LoRA 与决策头），按 Qwen3.5-4B 开源读出的同一方式。

本地生成 (样本, 守卫, 问题) 的 state 与陈述文本，GPU 机只做推理（base27b_run.py）。守卫：
  base27b_decomposed  MinGuard 的单事实问题 + max 融合
  base27b_holistic    单个整体问题
用法：FACTS_VERSION=1 python base27b_jobs.py   → runs/base27b_jobs.jsonl
"""

from __future__ import annotations

import json

import guards as G
import paper_data as P
import sqlfacts
from jev27b_jobs import items_to_judge

OUT = P.RUNS / "base27b_jobs.jsonl"


def main() -> None:
    items = items_to_judge()
    n = 0
    with open(OUT, "w") as fo:
        for i, x in items.items():
            f = sqlfacts.facts_for(x["db"], x["sql"])
            st = json.dumps(G.build_state(x, f, with_facts=True), ensure_ascii=False, indent=1)
            ind = bool(f.individual_level(x["db"]))
            rows = [("base27b_holistic", "violation", G.HOLISTIC)]
            rows += [("base27b_decomposed", k, q["instructions"]) for k, q in G.decomposed_questions(x, f).items()]
            for g, k, stmt in rows:
                fo.write(json.dumps({"id": i, "guard": g, "q": k, "individual": ind, "state": st,
                                     "statement": stmt}, ensure_ascii=False) + "\n")
                n += 1
    print(len(items), "requests,", n, "statements")


if __name__ == "__main__":
    main()
