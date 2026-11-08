# GPU 机上跑：27B 底座（不加 LoRA）的选项读出，与 readout_guard.py 相同的模板、双顺序平均与 p_yes。
# 用法：python base27b_run.py --model models/JEV-27B --jobs logs/base27b_jobs.jsonl --out runs/base27b_probs.jsonl
import argparse, json, math, os, time

from vllm import LLM, SamplingParams

TEMPLATE = """STATE:
{state}

STATEMENT:
{statement}

OPTIONS:
{options}
Select one option. Answer only with its bracketed numeric identifier."""
YES = "Yes, this statement is true for the given request."
NO = "No, this statement is false for the given request."

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--jobs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

jobs = [json.loads(l) for l in open(args.jobs)]
if args.limit:
    jobs = jobs[: args.limit]
done = set()
if os.path.exists(args.out):
    done = {(r["id"], r["guard"], r["q"]) for r in map(json.loads, open(args.out))}
todo = [j for j in jobs if (j["id"], j["guard"], j["q"]) not in done]

t0 = time.time()
llm = LLM(args.model, max_model_len=8192, gpu_memory_utilization=0.90, max_num_seqs=256,
          enable_prefix_caching=True, seed=0)
tok = llm.get_tokenizer()
print(f"[load] {time.time() - t0:.0f}s", flush=True)
id1 = tok.encode("1", add_special_tokens=False)[0]
id2 = tok.encode("2", add_special_tokens=False)[0]
sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)


def prompt(state, statement, opts):
    text = TEMPLATE.format(state=state, statement=statement,
                           options="\n".join(f"[{i + 1}] {o}" for i, o in enumerate(opts)))
    p = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return p + "Best answer: ["


def lp(o):
    d = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
    return {tid: v.logprob for tid, v in d.items()}


def p_yes(la, lb):
    def two(d):
        floor = min(d.values()) - 5.0 if d else -50.0
        return d.get(id1, floor), d.get(id2, floor)
    a1, a2 = two(la); b1, b2 = two(lb)
    return 0.5 * (1.0 / (1.0 + math.exp(a2 - a1)) + 1.0 / (1.0 + math.exp(b1 - b2)))


# 冒烟：一个显然为真的陈述应得到高概率
smoke = llm.generate([prompt('{"text": "The customer says the parcel arrived damaged and wants a refund."}',
                             "The customer is asking for a refund.", o) for o in ((YES, NO), (NO, YES))], sp,
                     use_tqdm=False)
print("[smoke] P(yes) =", round(p_yes(lp(smoke[0]), lp(smoke[1])), 4), flush=True)
print(f"[jobs] {len(jobs)} statements, {len(done)} done, {len(todo)} to run", flush=True)

BATCH = 1000
t = time.time()
with open(args.out, "a") as f:
    for k in range(0, len(todo), BATCH):
        part = todo[k:k + BATCH]
        ps = [p for j in part for p in (prompt(j["state"], j["statement"], (YES, NO)),
                                        prompt(j["state"], j["statement"], (NO, YES)))]
        outs = llm.generate(ps, sp, use_tqdm=False)
        for n, j in enumerate(part):
            f.write(json.dumps({"id": j["id"], "guard": j["guard"], "q": j["q"], "individual": j["individual"],
                                "p": p_yes(lp(outs[2 * n]), lp(outs[2 * n + 1]))}) + "\n")
        f.flush()
        dt = time.time() - t
        print(f"[batch] {k + len(part)}/{len(todo)}  {dt:.0f}s  {(k + len(part)) / dt:.1f} stmt/s", flush=True)
print("RUN_OK", flush=True)
