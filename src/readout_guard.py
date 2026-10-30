"""开源 Jev 式读出守卫：冻结的 Qwen3.5-4B 读选项标识符的对数概率（LLM-as-Jev 做法）。

与 Jev 守卫用同一份 state、同一组分解问题、同一个融合规则，只换判官，用来分开"分解问法"
和"Jev 本身"各自的贡献。每个问题按两种选项顺序各问一次取平均，抵消选项位置偏差。
用法（GPU 机）：python readout_guard.py --model ../models/Qwen3.5-4B --pairs a.jsonl:out_a.jsonl ...
"""

from __future__ import annotations

import argparse
import json
import math
import time

import guards as G
import sqlfacts

TEMPLATE = """STATE:
{state}

STATEMENT:
{statement}

OPTIONS:
{options}
Select one option. Answer only with its bracketed numeric identifier."""

YES = "Yes, this statement is true for the given request."
NO = "No, this statement is false for the given request."


def option_prompts(tok, state: dict, statement: str) -> list[str]:
    """两种顺序：[1]=是 / [1]=否。返回带 chat 模板的提示，以 'Best answer: [' 结尾。"""
    st = json.dumps(state, ensure_ascii=False, indent=1)
    out = []
    for opts in ((YES, NO), (NO, YES)):
        text = TEMPLATE.format(state=st, statement=statement,
                               options="\n".join(f"[{i + 1}] {o}" for i, o in enumerate(opts)))
        p = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        out.append(p + "Best answer: [")
    return out


def p_yes(lp_a: dict, lp_b: dict, id1: int, id2: int) -> float:
    """lp_*：首 token 的 {token_id: logprob}。顺序 a 里 1=是，顺序 b 里 2=是。"""
    def two(lp):
        floor = min(lp.values()) - 5.0 if lp else -50.0
        return lp.get(id1, floor), lp.get(id2, floor)
    a1, a2 = two(lp_a)
    b1, b2 = two(lp_b)
    pa = 1.0 / (1.0 + math.exp(a2 - a1))
    pb = 1.0 / (1.0 + math.exp(b1 - b2))
    return 0.5 * (pa + pb)


def score_items(llm, tok, items: list[dict]) -> list[dict]:
    """对每条样本跑整体（含事实）与分解两种问法，返回两条守卫结果。"""
    from vllm import SamplingParams
    id1 = tok.encode("1", add_special_tokens=False)[0]
    id2 = tok.encode("2", add_special_tokens=False)[0]
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    jobs = []   # (样本下标, 守卫名, 问题名, 提示 a, 提示 b)
    short, indiv = {}, {}
    for k, it in enumerate(items):
        f = sqlfacts.facts_for(it["db"], it["sql"])
        sc = G.short_circuit(it, f)
        if sc is not None:
            short[k] = sc
            continue
        indiv[k] = bool(f.individual_level(it["db"]))
        state = G.build_state(it, f, with_facts=True)
        a, b = option_prompts(tok, state, G.HOLISTIC)
        jobs.append((k, "readout_holistic", "violation", a, b))
        for name, q in G.decomposed_questions(it, f).items():
            a, b = option_prompts(tok, state, q["instructions"])
            jobs.append((k, "readout_decomposed", name, a, b))

    prompts = [p for j in jobs for p in (j[3], j[4])]
    t0 = time.time()
    outs = llm.generate(prompts, sp) if prompts else []
    dt = time.time() - t0

    def lp(o):
        d = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        return {tid: v.logprob for tid, v in d.items()}

    probs: dict[tuple[int, str], dict[str, float]] = dict()
    for i, j in enumerate(jobs):
        probs.setdefault((j[0], j[1]), {})[j[2]] = p_yes(lp(outs[2 * i]), lp(outs[2 * i + 1]), id1, id2)

    res = []
    per_ms = dt * 1000 / max(len(items) - len(short), 1)
    for k, it in enumerate(items):
        for g in ("readout_holistic", "readout_decomposed"):
            if k in short:
                s, why = short[k]
                res.append({"guard": g, "id": it["id"], "score": s, "why": why, "det": True, "latency_ms": 0.0})
                continue
            pr = probs.get((k, g), {})
            s, why = (pr.get("violation", 0.5), "violation") if g == "readout_holistic" else G.fuse(pr, individual=indiv[k])
            res.append({"guard": g, "id": it["id"], "score": s, "why": why, "probs": pr, "latency_ms": per_ms})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="基准:输出，冒号分隔")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85,
              enable_prefix_caching=True, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    tok = llm.get_tokenizer()
    for pair in args.pairs:
        bench, out = pair.split(":")
        items = [json.loads(l) for l in open(bench)]
        if args.limit:
            items = items[: args.limit]
        t0 = time.time()
        res = score_items(llm, tok, items)
        with open(out, "w") as fo:
            for r in res:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{bench}: {len(items)} 条，{time.time() - t0:.0f}s → {out}", flush=True)


if __name__ == "__main__":
    main()
