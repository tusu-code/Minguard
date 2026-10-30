# GPU 机上跑：JEV-27B System 1 离线推理。输入 jev27b_jobs.jsonl（本地生成），输出每个 prompt 的 P(true)。
# 读出方式与模型说明一致：只允许 false/true 两个 token，取 logprob，加决策头 bias，再除以 noul 温度。
# 用法：python jev27b_run.py --jobs logs/jev27b_jobs.jsonl --out runs/jev27b_probs.jsonl
import argparse, json, math, os, time

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/JEV-27B")
ap.add_argument("--jobs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
M = args.model
dh = json.load(open(f"{M}/adapter_vllm/decision_head.json"))
T = json.load(open(f"{M}/calibration.json"))["per_kind"]["noul"]
s0 = dh["slots"]["ranges"]["noul"][0]
IDS = dh["verbalizer_ids"][s0:s0 + 2]

jobs = [json.loads(l) for l in open(args.jobs)]
if args.limit:
    jobs = jobs[:args.limit]
t0 = time.time()
llm = LLM(M, enable_lora=True, max_lora_rank=32, logprobs_mode="processed_logprobs",
          max_model_len=8192, gpu_memory_utilization=0.90, max_num_seqs=256)
lora = LoRARequest("jev-decision", 1, f"{M}/adapter_vllm")
print(f"[load] {time.time() - t0:.0f}s", flush=True)

# 冒烟：模型说明里的例子，期望 P(true) ≈ 0.978
sp = SamplingParams(max_tokens=1, temperature=1.0, top_k=0, top_p=1.0, allowed_token_ids=IDS, logprobs=2)
def p_true(o):
    lp = {tid: v.logprob for tid, v in o.outputs[0].logprobs[0].items()}
    z = [(lp.get(t, -1e9) + dh["bias"][s0 + i]) / T for i, t in enumerate(IDS)]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    return e[1] / sum(e)
ex = ("[kind] noul\n[state] Customer says the parcel arrived damaged and wants their money back.\n"
      "[question] Is the customer asking for a refund?\n[options]\nfalse\ntrue\n[decision]:")
print("[smoke] P(true) =", round(p_true(llm.generate([ex], sp, lora_request=lora, use_tqdm=False)[0]), 4), flush=True)

# 分批推理、每批写盘：实例可能随时关机，重跑时跳过已完成的 (id, guard, q)
done = set()
if os.path.exists(args.out):
    for l in open(args.out):
        r = json.loads(l)
        done.add((r["id"], r["guard"], r["q"]))
todo = [j for j in jobs if (j["id"], j["guard"], j["q"]) not in done]
tok = llm.get_tokenizer()
print(f"[jobs] {len(jobs)} prompts, {len(done)} already done, {len(todo)} to run", flush=True)
BATCH = 2000
t, ntok = time.time(), 0
with open(args.out, "a") as f:
    for k in range(0, len(todo), BATCH):
        part = todo[k:k + BATCH]
        outs = llm.generate([j["prompt"] for j in part], sp, lora_request=lora, use_tqdm=False)
        for j, o in zip(part, outs):
            f.write(json.dumps({"id": j["id"], "guard": j["guard"], "q": j["q"], "individual": j["individual"],
                                "p": p_true(o)}) + "\n")
            ntok += len(o.prompt_token_ids)
        f.flush()
        dt = time.time() - t
        print(f"[batch] {k + len(part)}/{len(todo)}  {dt:.0f}s  {(k + len(part))/dt:.1f} q/s  "
              f"{ntok/dt/1e3:.1f}k tok/s", flush=True)
print("RUN_OK", flush=True)
