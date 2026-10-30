"""用本地 vLLM 让 Text-to-SQL 模型在含个人数据的 BIRD dev 库上生成 SQL（真实模型输出基准的原料）。

每题两种提示风格（omnisql 官方格式 / 朴素代理格式），贪心 1 条 + 温度采样 n 条。
输出 runs/gen_<model>.jsonl，每行一题一风格，含全部候选 SQL。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import dbmeta
import prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="本地模型目录")
    ap.add_argument("--name", required=True)
    ap.add_argument("--styles", default="omnisql,agent")
    ap.add_argument("--n", type=int, default=4, help="温度采样条数")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dbs", default=",".join(dbmeta.PILOT_DBS))
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    dbs = args.dbs.split(",")
    items = [x for x in json.load(open(dbmeta.BIRD_DEV)) if x["db_id"] in dbs]
    if args.limit:
        items = items[: args.limit]
    styles = args.styles.split(",")
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=12288, gpu_memory_utilization=0.90,
              enable_prefix_caching=True, seed=0)
    tok = llm.get_tokenizer()

    def chat(p: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True)

    reqs = [(x, s, chat(prompts.build(s, x))) for x in items for s in styles]
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    sample = SamplingParams(temperature=args.temperature, top_p=0.95, n=args.n, max_tokens=args.max_tokens,
                            seed=0)
    t0 = time.time()
    out_g = llm.generate([r[2] for r in reqs], greedy)
    out_s = llm.generate([r[2] for r in reqs], sample) if args.n else [None] * len(reqs)
    dt = time.time() - t0

    path = dbmeta.ROOT / f"runs/gen_{args.name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    n_tok = 0
    with open(path, "w") as fo:
        for (x, s, _), g, smp in zip(reqs, out_g, out_s):
            cands = [("greedy", g.outputs[0].text)]
            if smp is not None:
                cands += [(f"t{args.temperature}_{k}", o.text) for k, o in enumerate(smp.outputs)]
            n_tok += sum(len(o.token_ids) for o in g.outputs) + (
                sum(len(o.token_ids) for o in smp.outputs) if smp is not None else 0)
            fo.write(json.dumps({
                "qid": x["question_id"], "db": x["db_id"], "style": s, "model": args.name,
                "question": x["question"], "evidence": x["evidence"], "gold_sql": x["SQL"],
                "candidates": [{"tag": t, "sql": prompts.extract_sql(txt), "raw_len": len(txt)} for t, txt in cands],
            }, ensure_ascii=False) + "\n")
    print(f"{len(reqs)} 个请求，生成 {n_tok} token，用时 {dt:.0f}s → {path}")


if __name__ == "__main__":
    main()
