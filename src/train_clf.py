"""微调分类器基线：ModernBERT-base 读与守卫相同的 state 文本，二分类"应拦截"。

两种训练设定：
  src    只在试点基准上训练，到留出/真实基准上测（分布变化）
  src+cal 试点基准 + 目标基准的校准集一起训练（目标分布内，作为分类器的上界）
阈值和其他守卫一样由 evaluate.py 在目标校准集上定，输出格式与 run_guards 相同。
"""

from __future__ import annotations

import argparse
import json
import random
import time

import guards as G
import sqlfacts


def item_text(it: dict) -> str:
    f = sqlfacts.facts_for(it["db"], it["sql"])
    return json.dumps(G.build_state(it, f, with_facts=True), ensure_ascii=False)


def load(path: str, split: str | None = None) -> list[dict]:
    xs = [json.loads(l) for l in open(path)]
    return [x for x in xs if split is None or x["split"] == split]


def train(model_dir: str, texts: list[str], labels: list[int], epochs: int, lr: float,
          max_len: int, bs: int, seed: int):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(seed)
    random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    idx = list(range(len(texts)))
    steps = epochs * ((len(idx) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) *
                                              max(0.0, 1 - s / steps))
    model.train()
    for ep in range(epochs):
        random.shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            enc = tok([texts[k] for k in b], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to("cuda")
            y = torch.tensor([labels[k] for k in b], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**enc, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += float(loss) * len(b)
        print(f"  epoch {ep + 1}: loss {tot / len(idx):.4f}", flush=True)
    model.eval()
    return tok, model


def predict(tok, model, texts: list[str], max_len: int, bs: int = 32) -> list[float]:
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits.float()
            out.extend(torch.softmax(logits, -1)[:, 1].tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, help="训练用的试点基准")
    ap.add_argument("--targets", nargs="+", required=True, help="目标基准:输出，冒号分隔")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = load(args.source)
    src_x = [item_text(x) for x in src]
    src_y = [x["label"] for x in src]

    # 设定 1：只用试点训练一次，所有目标共用
    t0 = time.time()
    tok, model = train(args.model, src_x, src_y, args.epochs, args.lr, args.max_len, args.bs, args.seed)
    print(f"src 训练 {len(src)} 条，{time.time() - t0:.0f}s", flush=True)
    targets = []
    for pair in args.targets:
        bench, out = pair.split(":")
        items = load(bench)
        texts = [item_text(x) for x in items]
        targets.append((bench, out, items, texts))
        scores = predict(tok, model, texts, args.max_len)
        with open(out, "a") as fo:
            for it, s in zip(items, scores):
                fo.write(json.dumps({"guard": "clf_src", "id": it["id"], "score": s, "why": "clf",
                                     "latency_ms": 0.0}) + "\n")
    del model

    # 设定 2：每个目标单独加上它的校准集再训练（目标分布内的上界）
    for bench, out, items, texts in targets:
        cal = [(t, x["label"]) for t, x in zip(texts, items) if x["split"] == "cal"]
        tok, model = train(args.model, src_x + [t for t, _ in cal], src_y + [y for _, y in cal],
                           args.epochs, args.lr, args.max_len, args.bs, args.seed)
        scores = predict(tok, model, texts, args.max_len)
        with open(out, "a") as fo:
            for it, s in zip(items, scores):
                fo.write(json.dumps({"guard": "clf_src_cal", "id": it["id"], "score": s, "why": "clf",
                                     "latency_ms": 0.0}) + "\n")
        print(f"{bench}: src+cal 训练 {len(src_x) + len(cal)} 条，已写 {out}", flush=True)
        del model


if __name__ == "__main__":
    main()
