"""从打分文件直接生成论文表格（LaTeX），输出到 paper/tables/。数字一律由本脚本算出，不手抄。

用法：FACTS_VERSION=1 python make_tables.py
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score

import dbmeta
import evaluate as E
import paper_data as P

OUT = dbmeta.ROOT.parent / "paper" / "tables"
BENCH_NAMES = {"heldout": "Held-out", "real": "Real", "new_model": "New model", "new_db": "New DBs"}
GUARD_NAMES = {
    "jev_decomposed": r"\sys (decomposed)",
    "jev_holistic": "Jev, one question + facts",
    "jev_holistic_raw": "Jev, one question, no facts",
    "readout_decomposed": "Qwen3.5-4B readout, decomp.",
    "readout_holistic": "Qwen3.5-4B readout, one quest.",
    "llm_deepseek": "DeepSeek-V3.2 judge",
    "llm_qwen235b": "Qwen3-235B judge",
    "clf_src": "ModernBERT, dev data",
    "clf_src_cal": "ModernBERT, dev + target",
    "string_match": "Facts + keyword match",
    "llm_deepseek_decomposed": "DeepSeek-V3.2, decomp.",
    "base27b_decomposed": "Qwen3.8-27B, decomp.",
    "base27b_holistic": "Qwen3.8-27B, one quest.",
    "jev27b_decomposed": r"\sys, JEV-27B (open)",
    "jev27b_holistic": "JEV-27B, one quest. + facts",
}


def write(name: str, tex: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(tex)
    print(f"写出 {OUT / name}.tex")


def auc(items, S, g, ids) -> float:
    return roc_auc_score([items[i]["label"] for i in ids], [S[g][i]["score"] for i in ids])


def fmt(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".lstrip("0") if 0 <= x < 1 else f"{x:.{d}f}"


def pend(b: str, g: str, s: str) -> str:
    return rf"\textcolor{{red}}{{{s}}}" if (b, g) in P.PENDING else s


def tab_dbs() -> None:
    d = json.load(open(dbmeta.BIRD_DEV))
    rows = []
    for db in dbmeta.PILOT_DBS + dbmeta.HELDOUT_DBS:
        sch = dbmeta.load_schema(db)
        cats = Counter(c for t in sch for _, c in dbmeta.personal_columns(db, t))
        rows.append((db, len(sch), sum(len(v) for v in sch.values()), sum(cats.values()),
                     sum(1 for x in d if x["db_id"] == db), "design" if db in dbmeta.PILOT_DBS else "final"))
    short = {"thrombosis_prediction": "thrombosis", "student_club": "student\\_club", "financial": "financial",
             "codebase_community": "codebase", "formula_1": "formula\\_1", "european_football_2": "football",
             "california_schools": "ca\\_schools"}
    body = "\n".join(rf"\texttt{{{short[db]}}} & {t} & {c} & {p} & {q} & {u} \\"
                      for db, t, c, p, q, u in rows)
    tot = [sum(r[i] for r in rows) for i in (1, 2, 3, 4)]
    write("dbs", rf"""\begin{{table}}[t]
\caption{{BIRD databases with personal data. ``Personal'' counts columns marked with one of seven categories.
Design databases were used to develop \sys; final databases only in the final test.}}
\label{{tab:dbs}}
\centering\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}lrrrrl@{{}}}}
\toprule
Database & Tables & Cols. & Personal & Quest. & Use \\
\midrule
{body}
\midrule
Total & {tot[0]} & {tot[1]} & {tot[2]} & {tot[3]} & \\
\bottomrule
\end{{tabular}}
\end{{table}}
""")


WORKLOAD_DESC = {
    "pilot": ("Synthetic", "gold SQL + 3 mutation families", "4 design", "default, restrictive"),
    "heldout": ("Held-out", "5 unseen attack + 2 benign rewrite families", "4 design", "default, alt (unseen)"),
    "real": ("Real", "OmniSQL-7B, 2 prompts, greedy + 4 samples", "4 design", "default, restrictive"),
    "new_model": ("New model", "Qwen2.5-Coder-7B, frozen design", "4 design", "default, restrictive"),
    "new_db": ("New DBs", "OmniSQL-7B + Qwen2.5-Coder-7B, frozen design", "3 final", "default"),
}


def tab_workloads() -> None:
    rows = []
    for b, (name, src, dbs, pol) in WORKLOAD_DESC.items():
        items, S = P.load(b)
        J = P.judged(items, S)
        T = P.test_half(items, J)
        y = [items[i]["label"] for i in items]
        rows.append(rf"{name} & {src} & {dbs} & {pol} & {len(items)} & {sum(y)} & {len(J)} & {len(T)} \\")
    write("workloads", r"""\begin{table*}[t]
\caption{Workloads. ``Judged'' are requests the fact layer does not resolve; ``Test'' are judged requests whose
question has an odd BIRD identifier, which no trained baseline saw. All accuracy numbers in
Sec.~\ref{sec:exp} are on the test half.}
\label{tab:workloads}
\centering\footnotesize
\begin{tabular}{@{}llllrrrr@{}}
\toprule
Workload & Source of SQL & DBs & Policies & Requests & Positive & Judged & Test \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


MAIN_GUARDS = ["jev_decomposed", "jev27b_decomposed", "jev_holistic", "jev27b_holistic", "jev_holistic_raw",
               "base27b_decomposed", "base27b_holistic", "readout_decomposed", "readout_holistic",
               "llm_deepseek_decomposed", "llm_deepseek", "llm_qwen235b", "clf_src", "clf_src_cal", "string_match"]


def tab_main() -> None:
    data = {b: P.load(b) for b in BENCH_NAMES}
    tests = {b: P.test_half(it, P.judged(it, S)) for b, (it, S) in data.items()}
    best = {b: max(auc(it, S, g, tests[b]) for g in MAIN_GUARDS) for b, (it, S) in data.items()}
    lines = []
    for g in MAIN_GUARDS:
        cells = []
        for b, (it, S) in data.items():
            a = auc(it, S, g, tests[b])
            s = fmt(a)
            if abs(a - best[b]) < 5e-4:
                s = rf"\textbf{{{s}}}"
            cells.append(pend(b, g, s))
        lines.append(f"{GUARD_NAMES[g]} & " + " & ".join(cells) + r" \\")
        if g in ("jev27b_decomposed", "jev_holistic_raw", "readout_holistic", "llm_qwen235b", "clf_src_cal"):
            lines.append(r"\addlinespace")
    ns = " & ".join(f"{len(tests[b])}/{sum(data[b][0][i]['label'] for i in tests[b])}" for b in BENCH_NAMES)
    write("main", r"""\begin{table}[t]
\caption{AUROC on the test half of judged requests ($n$/positives below). Bold: best per column.}
\label{tab:main}
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Guard & """ + " & ".join(BENCH_NAMES.values()) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\midrule
$n$/pos. & """ + ns + r""" \\
\bottomrule
\end{tabular}
\end{table}
""")


def tab_diff() -> None:
    """主系统减各基线的 AUROC 差，配对 bootstrap 95% 区间，测试半边。"""
    data = {b: P.load(b) for b in BENCH_NAMES}
    tests = {b: P.test_half(it, P.judged(it, S)) for b, (it, S) in data.items()}
    lines = []
    for g in MAIN_GUARDS[1:]:
        cells = []
        for b, (it, S) in data.items():
            ids = tests[b]
            m, lo, hi, _ = E.boot_auc_diff_grouped(it, ids, S["jev_decomposed"], S[g])
            s = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            if lo > 0:
                s = rf"\textbf{{{s}}}"
            cells.append(pend(b, g, s))
        lines.append(f"{GUARD_NAMES[g]} & " + " & ".join(cells) + r" \\")
    write("diff", r"""\begin{table*}[t]
\caption{AUROC of \sys minus AUROC of each baseline on the test half, with 95\% intervals from a paired bootstrap that
resamples questions (2{,}000 resamples), so that candidates of the same question move together. Bold: interval excludes zero.}
\label{tab:diff}
\centering\footnotesize
\begin{tabular}{@{}lllll@{}}
\toprule
\sys minus & """ + " & ".join(BENCH_NAMES.values()) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


RISK_GUARDS = ["jev_decomposed", "jev27b_decomposed", "jev_holistic", "base27b_decomposed", "readout_decomposed",
               "llm_deepseek_decomposed",
               "llm_deepseek",
               "llm_qwen235b", "clf_src",
               "string_match"]


def tab_risk(reps: int = 200) -> None:
    """按题分组 conformal（α=5%、β=10%），200 次随机划分的均值与 90% 分位，判官子集。
    clf_src_cal 不在表里：它训练时用过一半题目，随机划分会把训练题放进测试侧。"""
    lines = []
    for b, name in BENCH_NAMES.items():
        it, S = P.load(b)
        J = P.judged(it, S)
        for k, g in enumerate(RISK_GUARDS):
            r = E.grouped_risk(it, S, g, J, 0.05, 0.10, reps=reps)
            cells = [f"{r[key][0]:.3f} / {r[key][1]:.3f}" for key in ("auto", "miss", "false_block")]
            cells = [pend(b, g, c) for c in cells]
            head = rf"\multirow{{{len(RISK_GUARDS)}}}{{*}}{{{name}}}" if k == 0 else ""
            lines.append(f"{head} & {GUARD_NAMES[g]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    write("risk", r"""\begin{table*}[t]
\caption{Question-grouped conformal risk control with miss budget $\alpha=5\%$ and false-block budget
$\beta=10\%$, over judged requests. Each cell is the mean / 90th percentile over 200 random splits of questions.
Automation is the fraction of judged requests allowed or blocked without escalation.}
\label{tab:risk}
\centering\footnotesize
\begin{tabular}{@{}llrrr@{}}
\toprule
Workload & Guard & Automated & Missed (of positives) & False blocks (of negatives) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")


# 消融：(显示名, 守卫名, 与主系统相比改了什么)
ABLATIONS = [
    ("Full system", "jev_decomposed", "--"),
    ("No facts, one question", "jev_holistic_raw", "remove fact layer and decomposition"),
    ("One question", "jev_holistic", "remove decomposition"),
    ("Open readout", "readout_decomposed", "replace Jev with Qwen3.5-4B readout"),
    ("Mean fusion", "fuse_mean", "average of violation terms instead of max"),
    ("Noisy-OR fusion", "fuse_noisyor", "$1-\\prod(1-v)$ over violation terms"),
    ("Learned fusion", "fuse_learned", "logistic regression on dev data"),
    ("Ungated fusion", "jev_decomposed_oldfuse", "aggregate\\_only fused unconditionally (pre-fix)"),
    ("Cacheable questions", "jev_cacheable", "one call per question, SQL-free states"),
    ("Row conditions (frozen)", "jev_decomposed_v2", "add row conditions; new scope question"),
]


def tab_ablation() -> None:
    data = {b: P.load(b) for b in BENCH_NAMES}
    tests = {b: P.test_half(it, P.judged(it, S)) for b, (it, S) in data.items()}
    lines = []
    for name, g, change in ABLATIONS:
        cells = []
        for b, (it, S) in data.items():
            if g not in S or any(i not in S[g] for i in tests[b]):
                cells.append("--")
                continue
            a = auc(it, S, g, tests[b])
            if g == "jev_decomposed":
                cells.append(fmt(a))
                continue
            m, lo, hi, _ = E.boot_auc_diff_grouped(it, tests[b], S[g], S["jev_decomposed"])
            s = f"{fmt(a)} ({m:+.3f})"
            if hi < 0:
                s = rf"\textit{{{s}}}"
            cells.append(pend(b, g, s))
        lines.append(f"{name} & {change} & " + " & ".join(cells) + r" \\")
    write("ablation", r"""\begin{table*}[t]
\caption{Ablations: AUROC on the test half, with the difference to the full system in parentheses. Italic: the
95\% question-clustered bootstrap interval of the difference lies below zero. ``Row conditions'' was designed, frozen and
then evaluated once on the final workloads; it is not used by \sys.}
\label{tab:ablation}
\centering\footnotesize
\begin{tabular}{@{}llllll@{}}
\toprule
Variant & Change & """ + " & ".join(BENCH_NAMES.values()) + r""" \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

    tab_eff()


def tab_eff() -> None:
    """效率：确定性层耗时、端到端吞吐与延迟、每千条成本、缓存。数字读自测量文件。"""
    import re
    lat = json.load(open(P.RUNS / "latency_natural.json"))
    thr = json.load(open(P.RUNS / "throughput.json"))
    n = lat["n"]
    allow = lat["short"]["allow"] / n
    stage = [("Parse + lineage", "parse"), ("Bounded dry run", "exec"), ("Small-aggregate count", "contrib"),
             ("Fact layer total", "total")]
    rows1 = "\n".join(rf"{name} & {lat[k]['50']:.1f} & {lat[k]['95']:.1f} & {lat[k]['99']:.1f} \\"
                      for name, k in stage)
    rows2 = "\n".join(
        rf"{r['concurrency']} & {r['qps']:.1f} & {r['total_p50']:.0f} & {r['total_p95']:.0f} & "
        rf"{r['total_p99'] / 1000:.1f}\,s \\" for r in thr.values())
    # 每千条请求的判官成本：自然分布下的送判比例 × 每次判官调用的平均价格（取自真实输出这次运行）
    def per_call(guard):
        rs = [json.loads(l) for l in open(P.RUNS / "scores_real.jsonl")]
        rs = [r for r in rs if r["guard"] == guard and not r.get("det")]
        return rs
    frac = lat["short"]["judge"] / n
    jev = per_call("jev_decomposed")
    jev_cost = sum(r.get("tokens", 0) for r in jev) / len(jev) * 0.042 / 1e6
    log = open(P.RUNS / "run_real.log").read()
    costs = {}
    for g in ("llm_deepseek", "llm_qwen235b"):
        m = re.search(rf"{g}：请求 (\d+)，缓存 \d+，token \d+\+\d+，\$([\d.]+)", log)
        costs[g] = float(m.group(2)) / int(m.group(1))
    lat_ms = {g: float(np.mean([r["latency_ms"] for r in per_call(g)])) for g in
              ("jev_decomposed", "llm_deepseek", "llm_qwen235b")}
    rows3 = "\n".join(rf"{name} & \${1000 * frac * c:.3f} & {lat['total']['50'] + frac * lat_ms[g]:.0f} \\"
                      for name, g, c in (("\\sys (Jev)", "jev_decomposed", jev_cost),
                                         ("DeepSeek-V3.2 judge", "llm_deepseek", costs["llm_deepseek"]),
                                         ("Qwen3-235B judge", "llm_qwen235b", costs["llm_qwen235b"])))
    write("eff", rf"""\begin{{table}}[t]
\caption{{Efficiency on the natural output distribution of OmniSQL-7B ({n} deduplicated requests; the fact
layer allows {100 * allow:.1f}\% outright). (a) Fact-layer latency in ms. (b) End-to-end guard throughput and
latency in ms on 400 sampled requests, Jev calls uncached. (c) Judge cost and mean latency per request, with the
judged fraction of {100 * frac:.1f}\%.}}
\label{{tab:eff}}
\centering\footnotesize
\begin{{tabular}}{{@{{}}lrrr@{{}}}}
\multicolumn{{4}}{{@{{}}l}}{{(a) Fact layer}} \\
\toprule
Stage & p50 & p95 & p99 \\
\midrule
{rows1}
\bottomrule
\end{{tabular}}\\[4pt]
\begin{{tabular}}{{@{{}}rrrrr@{{}}}}
\multicolumn{{5}}{{@{{}}l}}{{(b) End-to-end, by client concurrency}} \\
\toprule
Conc. & Req./s & p50 & p95 & p99 \\
\midrule
{rows2}
\bottomrule
\end{{tabular}}\\[4pt]
\begin{{tabular}}{{@{{}}lrr@{{}}}}
\multicolumn{{3}}{{@{{}}l}}{{(c) Judge cost per 1000 requests and mean latency (ms)}} \\
\toprule
Judge & Cost & Latency \\
\midrule
{rows3}
\bottomrule
\end{{tabular}}
\end{{table}}
""")



HUMAN_CORE = {"多了个人属性列", "统计量改成逐条列出", "范围变大（多了无关的人）", "只多了编号"}


def human_eval():
    """人工盲标 200 条：返回 (items, S)。歧义类型（多给筛选值、取错人、未去重）不计入。"""
    recs = json.load(open(dbmeta.ROOT / "annotation/human_labels.json"))
    _, S = P.load("real_full")
    extra = [P.RUNS / "scores_human_extra.jsonl", P.RUNS / "scores_string.jsonl", P.RUNS / "scores_jev27b.jsonl",
             P.RUNS / "scores_llm_decomp.jsonl", P.RUNS / "scores_base27b.jsonl"]
    for line in (l for p in extra if p.exists() for l in open(p)):
        r = json.loads(line)
        S.setdefault(r["guard"], {})[r["id"]] = r
    P._fill_det(S, ["jev27b_decomposed", "jev27b_holistic", "llm_deepseek_decomposed", "base27b_decomposed",
                    "base27b_holistic"])   # 补充样本里被事实层直接判定的
    raw = {json.loads(l)["id"]: json.loads(l) for l in open(P.DATA / "real_bench_raw.jsonl")}
    items = {}
    for r in recs:
        if r["verdict"] == "不越量":
            y = 0
        elif r["type"] in HUMAN_CORE:
            y = 1
        else:
            continue
        items[r["id"]] = {**raw[r["id"]], "label": y}
    return recs, items, S


def human_design():
    """设计库 200 条、两位标注人独立盲标：返回 (items, S, labels)，labels[k] 为 {id: 0/1}，k ∈ ann1/ann2/consensus。
    歧义类型（多给筛选值、取错人、其他）和"无法判断"不计入；S 含补充样本的全部守卫打分。"""
    recs, _, S = human_eval()
    P._fill_det(S, ["jev27b_decomposed", "jev27b_holistic", "llm_deepseek_decomposed", "base27b_decomposed",
                    "base27b_holistic"])
    raw = {json.loads(l)["id"]: json.loads(l) for l in open(P.DATA / "real_bench_raw.jsonl")}
    def lab(v, t):
        return 0 if v == "不越量" else (1 if v == "越量" and t in HUMAN_CORE else None)
    L = {"ann1": {}, "ann2": {}}
    for r in recs:
        for k, (v, t) in {"ann1": (r["verdict"], r["type"]), "ann2": (r.get("verdict2"), r.get("type2"))}.items():
            y = lab(v, t)
            if y is not None:
                L[k][r["id"]] = y
    L["consensus"] = {i: y for i, y in L["ann1"].items() if L["ann2"].get(i) == y}
    items = {i: raw[i] for i in raw if any(i in L[k] for k in L)}
    return items, S, L


def tab_human() -> None:
    it, S, L = human_design()
    def sub_of(lab, kind):
        ids = list(lab)
        if kind in ("dag", "ddag"):
            ids = [i for i in ids if i in S["readout_decomposed"] and i in S["clf_src"]]
        if kind == "ddag":
            ids = [i for i in ids if it[i]["split"] == "test"]
        return ids
    def auc_on(g, lab, ids):
        return roc_auc_score([lab[i] for i in ids], [S[g][i]["score"] for i in ids])
    lines = []
    def row(g, kind="", note=""):
        cons = L["consensus"]; ids = sub_of(cons, kind)
        a = auc_on(g, cons, ids)
        if g == "jev_decomposed":
            d = "--"
        else:
            items = {i: {**it[i], "label": cons[i]} for i in ids}
            m, lo, hi, _ = E.boot_auc_diff_grouped(items, ids, S["jev_decomposed"], S[g])
            d = f"{m:+.3f} [{lo:+.2f}, {hi:+.2f}]".replace("0.", ".")
            if lo > 0:
                d = rf"\textbf{{{d}}}"
        a1 = auc_on(g, L["ann1"], sub_of(L["ann1"], kind)); a2 = auc_on(g, L["ann2"], sub_of(L["ann2"], kind))
        lines.append(rf"{GUARD_NAMES[g]}{note} & {fmt(a)} & {d} & {fmt(a1)}/{fmt(a2)} \\")
    for g in ["jev_decomposed", "jev27b_decomposed", "jev_holistic", "jev_holistic_raw", "base27b_decomposed",
              "llm_deepseek_decomposed", "llm_deepseek", "llm_qwen235b", "string_match"]:
        row(g)
        if g in ("jev27b_decomposed", "jev_holistic_raw", "llm_qwen235b"):
            lines.append(r"\addlinespace")
    lines.append(r"\addlinespace")
    for g in ["jev_decomposed", "readout_decomposed", "clf_src"]:
        row(g, "dag", r"$^\dagger$")
    lines.append(r"\addlinespace")
    for g in ["jev_decomposed", "clf_src_cal"]:
        row(g, "ddag", r"$^\ddagger$")
    def np_(lab, kind):
        ids = sub_of(lab, kind); return f"{len(ids)}/{sum(lab[i] for i in ids)}"
    c, a1, a2 = L["consensus"], L["ann1"], L["ann2"]
    write("human", rf"""\begin{{table}}[t]
\caption{{AUROC against two independent blind annotations of 200 Real requests. Cons.: the {len(c)} requests
({sum(c.values())} over-disclosing) both annotators label unambiguously and identically, with the difference of \sys
to each guard and a question-clustered bootstrap 95\% interval. Ann.~1/2: each annotator's own unambiguous labels
($n$/positives {np_(a1, '')} and {np_(a2, '')}). $^\dagger$Requests all guards scored ({np_(c, 'dag')} under the
consensus). $^\ddagger$Of those, the test half ({np_(c, 'ddag')}), which the target-label classifier did not train
on.}}
\label{{tab:human}}
\centering\footnotesize
\setlength{{\tabcolsep}}{{2pt}}
\begin{{tabular}}{{@{{}}lrlr@{{}}}}
\toprule
Guard & Cons. & \sys minus guard & Ann.~1/2 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def human_newdb():
    """新库 150 条、两位标注人独立盲标：返回 (items, S, labels)，labels[k] 为 {id: 0/1}，k ∈ ann1/ann2/consensus。
    只用在评测子集内（有全部守卫打分）的请求；歧义类型（多给筛选值、取错人、其他）不计入。"""
    # 全部 211 条测试半边判官请求：早先 150 条样本里落在测试半边的 46 条 + 补标的 165 条（merge_newdb_test.py）
    it, S = P.load("new_db")
    test = set(P.test_half(it, P.judged(it, S)))
    recs = [r for r in json.load(open(dbmeta.ROOT / "annotation/human_labels_newdb.json")) if r["id"] in test]
    recs += json.load(open(dbmeta.ROOT / "annotation/human_labels_newdb_test.json"))
    def lab(v, t):
        return 0 if v == "不越量" else (1 if t in HUMAN_CORE else None)
    L = {"ann1": {}, "ann2": {}}
    for r in recs:
        if r["id"] not in it:
            continue
        for j in (1, 2):
            y = lab(r[f"verdict{j}"], r[f"type{j}"])
            if y is not None:
                L[f"ann{j}"][r["id"]] = y
    L["consensus"] = {i: y for i, y in L["ann1"].items() if L["ann2"].get(i) == y}
    # 第三位标注者盲标仲裁后的最终标签（merge_adjudication.py；多数规则见 make_annotation_adjudication.py）
    adj = json.load(open(dbmeta.ROOT / "annotation/human_labels_newdb_adjudicated.json"))
    L["final"] = {r["id"]: r["final"] for r in adj if r["final"] is not None}
    return recs, it, S, L


def tab_human_newdb() -> None:
    recs, it, S, L = human_newdb()
    def aucs(g, lab):
        ids = list(lab)
        return roc_auc_score([lab[i] for i in ids], [S[g][i]["score"] for i in ids])
    cons = L["final"]
    ids = list(cons)
    items = {i: {**it[i], "label": cons[i]} for i in ids}
    lines = []
    for g in ["jev_decomposed", "jev27b_decomposed", "jev_holistic", "jev_holistic_raw", "base27b_decomposed",
              "readout_decomposed", "llm_deepseek_decomposed", "llm_deepseek", "llm_qwen235b", "clf_src", "clf_src_cal",
              "string_match"]:
        a = aucs(g, cons)
        if g == "jev_decomposed":
            d = "--"
        else:
            m, lo, hi, _ = E.boot_auc_diff_grouped(items, ids, S["jev_decomposed"], S[g])
            d = f"{m:+.3f} [{lo:+.2f}, {hi:+.2f}]".replace("0.", ".")
            if lo > 0:
                d = rf"\textbf{{{d}}}"
        lines.append(rf"{GUARD_NAMES[g]} & {fmt(a)} & {d} & {fmt(aucs(g, L['ann1']))}/{fmt(aucs(g, L['ann2']))} \\")
        if g in ("jev27b_decomposed", "jev_holistic_raw", "llm_qwen235b"):
            lines.append(r"\addlinespace")
    auto = roc_auc_score([it[i]["label"] for i in ids], [S["jev_decomposed"][i]["score"] for i in ids])
    npos = sum(cons.values())
    n1, n2 = (f"{len(L[k])}/{sum(L[k].values())}" for k in ("ann1", "ann2"))
    write("human_newdb", rf"""\begin{{table}}[t]
\caption{{AUROC against blind human labels of all judged New DBs requests in the test half. Final: the {len(ids)} requests
({npos} over-disclosing) with a label after a third annotator resolved the cases the first two do not agree on, with the
difference of \sys to each guard and a question-clustered bootstrap 95\% interval. Ann.~1/2: the first two annotators' own
unambiguous labels ($n$/positives {n1} and {n2}).}}
\label{{tab:human-newdb}}
\centering\footnotesize
\setlength{{\tabcolsep}}{{2pt}}
\begin{{tabular}}{{@{{}}lrlr@{{}}}}
\toprule
Guard & Final & \sys minus guard & Ann.~1/2 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


def tab_budget() -> None:
    """标注量曲线（label_budget.py 的结果）：校准题目数 n 与自动化率、校准侧平均正例数。"""
    res = json.load(open(P.RUNS / "label_budget.json"))
    rows = []
    for n in (25, 50, 100, 200):
        cells = []
        for b in BENCH_NAMES:
            r = res.get(f"{b}|jev_decomposed|{n}")
            cells.append("--" if r is None else f"{r['auto']:.2f} ({r['n_pos_cal']:.0f})")
        rows.append(f"{n} & " + " & ".join(cells) + r" \\")
    over = [res[k] for k in res if k.split("|")[1] == "jev_decomposed" and int(k.split("|")[2]) >= 100]
    om = max(r["over_miss"] for r in over)
    ofb = max(r["over_fb"] for r in over)
    write("budget", rf"""\begin{{table}}[t]
\caption{{Labels needed for risk control. Automation of \sys's judged requests when $n$ labeled questions calibrate
the thresholds ($\alpha=5\%$, $\beta=10\%$, mean over 500 splits); in parentheses the mean number of positives
among the $n$ calibration requests. New DBs has 104 judged questions. For $n\ge100$, single splits exceed the miss
budget in up to {om:.0%} and the false-block budget in up to {ofb:.0%} of splits; the means stay within budget.}}
\label{{tab:budget}}
\centering\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}rllll@{{}}}}
\toprule
$n$ & """ + " & ".join(BENCH_NAMES.values()) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""".replace("%", "\\%") if False else rf"""\begin{{table}}[t]
\caption{{Labels needed for risk control. Automation of \sys's judged requests when $n$ labeled questions calibrate
the thresholds ($\alpha=5\%$, $\beta=10\%$, mean over 500 splits); in parentheses the mean number of positives
among the $n$ calibration requests. New DBs has 104 judged questions. For $n\ge100$, single splits exceed the miss
budget in up to {om * 100:.0f}\% and the false-block budget in up to {ofb * 100:.0f}\% of splits; the means stay
within budget.}}
\label{{tab:budget}}
\centering\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{@{{}}rllll@{{}}}}
\toprule
$n$ & """ + " & ".join(BENCH_NAMES.values()) + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")


if __name__ == "__main__":
    tab_dbs()
    tab_workloads()
    tab_main()
    tab_diff()
    tab_ablation()
    tab_risk()
    tab_eff()
    tab_human()
    tab_human_newdb()
    tab_budget()
