"""试点评测：区分度（AUROC）、校准（ECE）、风险受控下的自动化率，按类别拆解。

风险控制（split conformal，校准集定阈值，测试集报实际值）：
  s < l 自动放行，s ≥ h 自动拦截，其余转人工。
  l 取校准集上满足 (漏放正例数 + 1)/(正例数 + 1) ≤ α 的最大值；
  h 取满足 (误拦负例数 + 1)/(负例数 + 1) ≤ β 的最小值；l > h 时合并为单阈值。
  自动化率 = 测试集上未转人工的比例。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

import dbmeta

BENCH = dbmeta.ROOT / "data/pilot_bench.jsonl"
SCORES = dbmeta.ROOT / "runs/scores.jsonl"


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (p >= lo) & (p < lo + 1 / bins) if lo < 0.9 else (p >= lo)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def thresholds(y, s, alpha, beta):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    cands = np.unique(np.concatenate([s, [0.0, 1.0 + 1e-9]]))
    l = 0.0
    for t in cands:  # 放行 s < t
        if ((pos < t).sum() + 1) / (len(pos) + 1) <= alpha:
            l = t
    h = 1.0 + 1e-9
    for t in cands[::-1]:  # 拦截 s >= t
        if ((neg >= t).sum() + 1) / (len(neg) + 1) <= beta:
            h = t
    if l > h:
        # 放行区与拦截区重叠时重叠部分判拦截：放行区只缩不扩，漏放上限仍成立；拦截区不变，误拦上限仍成立
        l = h
    return l, h


def operate(y, s, l, h):
    y, s = np.asarray(y), np.asarray(s)
    allow, block = s < l, s >= h
    esc = ~(allow | block)
    pos, neg = y == 1, y == 0
    return {"auto": float((~esc).mean()), "miss": float((allow & pos).sum() / max(pos.sum(), 1)),
            "false_block": float((block & neg).sum() / max(neg.sum(), 1)),
            "escalate": float(esc.mean()), "auto_allow": float(allow.mean()), "auto_block": float(block.mean())}


def boot_auc_diff(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, a, b))
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.10)
    ap.add_argument("--scores", default=str(SCORES))
    ap.add_argument("--bench", default=str(BENCH))
    ap.add_argument("--grouped", type=int, default=0, help="按题分组 conformal 的重复次数（0 不做）")
    ap.add_argument("--block", default="", help="只评测该 block 的样本（最终测试集用）")
    ap.add_argument("--json", default=str(dbmeta.ROOT / "runs/summary.json"))
    args = ap.parse_args()
    items = {json.loads(l)["id"]: json.loads(l) for l in open(args.bench)}
    if args.block:
        items = {k: v for k, v in items.items() if v.get("block") == args.block}
    S = defaultdict(dict)
    for l in open(args.scores):
        r = json.loads(l)
        S[r["guard"]][r["id"]] = r
    guards = sorted(S)
    common = set(items)
    for g in guards:
        common &= set(S[g])
    ids = sorted(common)
    judged = [i for i in ids if not S["jev_decomposed"][i].get("det")]
    print(f"共 {len(ids)} 条（全部守卫都有结果）；进入判官的 {len(judged)} 条，其余由确定性层短路")
    summary = {"n": len(ids), "n_judged": len(judged), "alpha": args.alpha, "beta": args.beta, "guards": {}}

    for subset_name, sub in (("全部", ids), ("仅判官子集", judged)):
        print(f"\n===== {subset_name}（α={args.alpha} 漏放上限，β={args.beta} 误拦上限）")
        print(f"{'守卫':18s} {'AUROC':>6s} {'ECE':>6s} | {'自动化':>6s} {'漏放':>6s} {'误拦':>6s} {'转人工':>6s}")
        for g in guards:
            cal = [i for i in sub if items[i]["split"] == "cal"]
            te = [i for i in sub if items[i]["split"] == "test"]
            yc = [items[i]["label"] for i in cal]; sc = [S[g][i]["score"] for i in cal]
            yt = [items[i]["label"] for i in te]; st = [S[g][i]["score"] for i in te]
            ya = [items[i]["label"] for i in sub]; sa = [S[g][i]["score"] for i in sub]
            auc = roc_auc_score(ya, sa) if len(set(sa)) > 1 else 0.5
            e = ece(ya, sa)
            l, h = thresholds(yc, sc, args.alpha, args.beta)
            op = operate(yt, st, l, h)
            print(f"{g:18s} {auc:6.3f} {e:6.3f} | {op['auto']:6.3f} {op['miss']:6.3f} {op['false_block']:6.3f} "
                  f"{op['escalate']:6.3f}   (l={l:.3f}, h={h:.3f})")
            summary["guards"].setdefault(g, {})[subset_name] = {"auroc": auc, "ece": e, "l": l, "h": h, **op}

    if args.grouped:
        print(f"\n===== 按题分组 conformal（{args.grouped} 次随机划分，均值 / 90% 分位；仅判官子集）")
        print(f"{'守卫':18s} {'自动化':>14s} {'漏放':>14s} {'误拦':>14s}")
        for g in guards:
            r = grouped_risk(items, S, g, judged, args.alpha, args.beta, reps=args.grouped)
            summary["guards"][g]["grouped"] = r
            print(f"{g:18s} " + " ".join(f"{r[k][0]:6.3f}/{r[k][1]:6.3f}" for k in ("auto", "miss", "false_block")))

    print("\n===== 配对 bootstrap：AUROC 差（仅判官子集）")
    y = [items[i]["label"] for i in judged]
    for a, b in (("jev_decomposed", "jev_holistic"), ("jev_decomposed", "jev_holistic_raw"),
                 ("jev_holistic", "jev_holistic_raw"), ("jev_decomposed", "llm_qwen235b"),
                 ("jev_decomposed", "llm_deepseek"), ("readout_decomposed", "readout_holistic"),
                 ("jev_decomposed", "readout_decomposed"), ("jev_decomposed", "clf_src"),
                 ("jev_decomposed", "clf_src_cal"), ("jev_decomposed", "jev_decomposed_v1"),
                 ("jev_cacheable", "jev_decomposed"), ("jev_cacheable", "jev_holistic")):
        if a in S and b in S:
            m, lo, hi, p = boot_auc_diff(y, [S[a][i]["score"] for i in judged], [S[b][i]["score"] for i in judged])
            print(f"  {a} − {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  P(≤0)={p:.3f}")

    print("\n===== 按类别：判为违规（s≥0.5）的比例")
    cats = sorted({items[i]["category"] for i in ids})
    print(f"{'类别':5s} {'标签':>4s} {'n':>4s} " + " ".join(f"{g[:12]:>12s}" for g in guards))
    for c in cats:
        cid = [i for i in ids if items[i]["category"] == c]
        lab = items[cid[0]]["label"]
        print(f"{c:5s} {lab:4d} {len(cid):4d} " + " ".join(
            f"{np.mean([S[g][i]['score'] >= 0.5 for i in cid]):12.2f}" for g in guards))

    print("\n===== 成本与延迟（进入判官的样本）")
    for g in guards:
        toks = [S[g][i].get("tokens", 0) for i in judged]
        lat = [S[g][i]["latency_ms"] for i in judged]
        print(f"  {g:18s} 平均输入 {np.mean(toks):7.0f} token  延迟 p50 {np.percentile(lat, 50):7.0f} ms "
              f"p95 {np.percentile(lat, 95):7.0f} ms")
    json.dump(summary, open(args.json, "w"), ensure_ascii=False, indent=1)



def grouped_risk(items: dict, S: dict, guard: str, ids: list[str], alpha: float, beta: float,
                 reps: int = 200, seed: int = 0) -> dict:
    """按题分组的 split conformal：题号随机对半分，校准侧每题只随机取 1 条（题间独立），
    测试侧每条按 1/该题条数加权（对应"随机抽一道题、再随机抽一条候选"的保证对象）。
    返回多次划分的平均值与 90% 分位数。"""
    rng = np.random.default_rng(seed)
    by_q: dict = defaultdict(list)
    for i in ids:
        by_q[(items[i]["db"], items[i]["qid"], items[i]["policy"])].append(i)
    qs = list(by_q)
    res = defaultdict(list)
    for _ in range(reps):
        perm = rng.permutation(len(qs))
        cal_q, te_q = [qs[k] for k in perm[: len(qs) // 2]], [qs[k] for k in perm[len(qs) // 2:]]
        cal = [by_q[q][rng.integers(len(by_q[q]))] for q in cal_q]
        l, h = thresholds([items[i]["label"] for i in cal], [S[guard][i]["score"] for i in cal], alpha, beta)
        wpos = wneg = mp = fb = auto = allow = block = wall = 0.0
        for q in te_q:
            w = 1.0 / len(by_q[q])
            for i in by_q[q]:
                s, y = S[guard][i]["score"], items[i]["label"]
                wall += w
                auto += w * (s < l or s >= h)
                allow += w * (s < l)
                block += w * (s >= h)
                if y == 1:
                    wpos += w; mp += w * (s < l)
                else:
                    wneg += w; fb += w * (s >= h)
        res["auto"].append(auto / wall); res["miss"].append(mp / max(wpos, 1e-9))
        res["allow"].append(allow / wall); res["block"].append(block / wall)
        res["false_block"].append(fb / max(wneg, 1e-9))
    return {k: (float(np.mean(v)), float(np.percentile(v, 90))) for k, v in res.items()}


if __name__ == "__main__":
    main()


def boot_auc_diff_grouped(items: dict, ids: list[str], a: dict, b: dict, n: int = 2000, seed: int = 0):
    """按题聚类的配对 bootstrap：有放回地抽题，题内全部请求一起进入；同题多候选不再被当作独立样本。"""
    rng = np.random.default_rng(seed)
    groups: dict = defaultdict(list)
    for i in ids:
        groups[(items[i]["db"], items[i]["qid"])].append(i)
    keys = list(groups)
    y_all = {i: items[i]["label"] for i in ids}
    d = []
    for _ in range(n):
        pick = [i for k in rng.integers(0, len(keys), len(keys)) for i in groups[keys[k]]]
        y = [y_all[i] for i in pick]
        if min(y) == max(y):
            continue
        d.append(roc_auc_score(y, [a[i]["score"] for i in pick]) - roc_auc_score(y, [b[i]["score"] for i in pick]))
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())
