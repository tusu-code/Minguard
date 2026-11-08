"""论文表格用的数据装载：每个基准 × 每个守卫的打分。

所有守卫统一用 v1 事实（与主系统一致）。
jev_holistic_raw 与两条规则不读事实，v1/v2 结果相同，直接用 v2 那次运行的打分。
"""

from __future__ import annotations

import json
from collections import defaultdict

import dbmeta
import guards as G

RUNS = dbmeta.ROOT / "runs"
DATA = dbmeta.ROOT / "data"
BENCHES = ["heldout", "real", "new_model", "new_db"]
PENDING: set = set()   # 最终测试集上的开源读出与分类器已用 v1 事实重跑（readout_final_v1、clf_final_v1）


def _read(path, keep=None, rename=None) -> dict:
    out: dict = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if keep is not None and r["guard"] not in keep:
            continue
        out[(rename or {}).get(r["guard"], r["guard"])][r["id"]] = r
    return out


def _merge(*parts) -> dict:
    S: dict = defaultdict(dict)
    for p in parts:
        for g, d in p.items():
            S[g].update(d)
    return S


def _items(path, block=None) -> dict:
    xs = [json.loads(l) for l in open(path)]
    return {x["id"]: x for x in xs if block is None or x.get("block") == block}


def _add_old_fusion(S: dict) -> None:
    """派生变体：修正前的融合规则（aggregate_only 无条件参与融合），用已存的问题概率重算。"""
    old = {}
    for i, r in S["jev_decomposed"].items():
        if r.get("det") or not r.get("probs"):
            old[i] = r
        else:
            s, why = G.fuse(r["probs"], individual=True)
            old[i] = {**r, "score": s, "why": why}
    S["jev_decomposed_oldfuse"] = old


def load(bench: str) -> tuple[dict, dict]:
    if bench == "pilot":
        items = _items(DATA / "pilot_bench.jsonl")
        S = _read(RUNS / "fixed_pilot.jsonl")
    elif bench == "heldout":
        items = _items(DATA / "heldout_bench.jsonl")
        S = _merge(_read(RUNS / "fixed_heldout.jsonl"), _read(RUNS / "scores_cacheable_heldout.jsonl"))
    elif bench in ("real", "real_full"):
        items = _items(DATA / ("real_bench_strict.jsonl" if bench == "real" else "real_bench.jsonl"))
        S = _merge(_read(RUNS / "fixed_real.jsonl"), _read(RUNS / "scores_cacheable_real.jsonl"))
    elif bench in ("new_model", "new_db"):
        items = _items(DATA / "final_bench.jsonl", block=bench)
        S = _merge(
            _read(RUNS / "scores_final_v2.jsonl", keep={"jev_holistic_raw", "rule_personal", "rule_acl"}),
            _read(RUNS / "scores_final_v2.jsonl", keep={"jev_decomposed"},
                  rename={"jev_decomposed": "jev_decomposed_v2"}),
            _read(RUNS / "scores_final_v1base.jsonl"),
            _read(RUNS / "scores_final_v1.jsonl"),
            _read(RUNS / "readout_final_v1.jsonl"),
            _read(RUNS / "clf_final_v1.jsonl"),
            _read(RUNS / "scores_cacheable_final.jsonl"))
    else:
        raise ValueError(bench)
    S = _merge(S, _read(RUNS / "scores_string.jsonl"))   # 审稿意见补的字符串匹配基线（string_baseline.py）
    if (RUNS / "scores_fusion.jsonl").exists():          # 融合方式消融（fusion_ablation.py）
        S = _merge(S, _read(RUNS / "scores_fusion.jsonl"))
        _fill_det(S, ["fuse_max", "fuse_mean", "fuse_noisyor", "fuse_learned"])
    if (RUNS / "scores_llm_decomp.jsonl").exists():      # 审稿意见补的 DeepSeek-V3.2 + 单事实问题（llm_decomposed.py）
        S = _merge(S, _read(RUNS / "scores_llm_decomp.jsonl"))
        _fill_det(S, ["llm_deepseek_decomposed"])
    if (RUNS / "scores_base27b.jsonl").exists():         # 审稿意见补的独立开源基线：27B 底座、不加 Jev 适配器（base27b_score.py）
        S = _merge(S, _read(RUNS / "scores_base27b.jsonl"))
        _fill_det(S, ["base27b_decomposed", "base27b_holistic"])
    if (RUNS / "scores_jev27b.jsonl").exists():          # 开源复现 JEV-27B（jev27b_score.py）
        S = _merge(S, _read(RUNS / "scores_jev27b.jsonl"))
        _fill_det(S, ["jev27b_decomposed", "jev27b_holistic"])
    S = {g: {i: r for i, r in d.items() if i in items} for g, d in S.items()}
    _add_old_fusion(S)
    return items, S


def _fill_det(S: dict, guards: list[str]) -> None:
    """事实层直接判定的请求不送判官，沿用主系统的确定性结果（两者事实层相同）。"""
    for g in guards:
        d = S.setdefault(g, {})
        for i, r in S["jev_decomposed"].items():
            if r.get("det") and i not in d:
                d[i] = r


def judged(items: dict, S: dict) -> list[str]:
    """进入判官的样本（主系统没有被确定性层直接判定的）。"""
    ours = S["jev_decomposed"]
    return [i for i in items if i in ours and not ours[i].get("det")]


def test_half(items: dict, ids: list[str]) -> list[str]:
    """题号为奇数的一半：分类器 clf_src_cal 训练时没见过。"""
    return [i for i in ids if items[i]["split"] == "test"]
