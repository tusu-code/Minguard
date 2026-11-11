"""论文的三张数据图（与 Fig. 1 同一套字体和配色），输出到 paper/figures/：
  fig_diff.pdf    各基线与 MinGuard 的 AUROC 差值和 95% 区间（替代原 Table V）
  fig_risk.pdf    风险控制下放行、拦截、转人工的比例，以及平均误放率和误拦率（替代原 Table VI）
  fig_budget.pdf  校准题目数与自动化率（替代原 Table XI）
数字先算好存进 runs/fig_data.json；加 --recompute 重算（bootstrap 与 200 次划分，约 20 分钟）。
用法：FACTS_VERSION=1 python make_figures.py [--recompute]
"""

from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

import dbmeta
import evaluate as E
import make_tables as M
import paper_data as P

OUT = dbmeta.ROOT.parent / "paper" / "figures"
FONTS = OUT / "fonts"
CACHE = P.RUNS / "fig_data.json"
C = dict(blue="#2F6FD0", orange="#E07B2E", purple="#7353BA", green="#2E9B5B", red="#D2453A", gray="#5B6670",
         light="#EEF0F3")
SHORT = {
    "jev27b_decomposed": "MinGuard with JEV-27B", "jev_holistic": "Jev, one question",
    "jev27b_holistic": "JEV-27B, one question", "jev_holistic_raw": "Jev, one quest., no facts",
    "readout_decomposed": "Qwen3.5-4B readout", "readout_holistic": "Qwen3.5-4B, one quest.",
    "llm_deepseek_decomposed": "DeepSeek-V3.2, decomp.",
    "base27b_decomposed": "Qwen3.8-27B readout", "base27b_holistic": "Qwen3.8-27B, one quest.",
    "llm_deepseek": "DeepSeek-V3.2 judge", "llm_qwen235b": "Qwen3-235B judge",
    "clf_src": "ModernBERT, dev", "clf_src_cal": "ModernBERT, dev + target",
    "string_match": "Keyword rule", "jev_decomposed": "MinGuard",
}
GROUP = {"jev27b_decomposed": C["blue"], "jev_holistic": C["blue"], "jev27b_holistic": C["blue"],
         "jev_holistic_raw": C["blue"], "readout_decomposed": C["purple"], "readout_holistic": C["purple"],
         "base27b_decomposed": C["purple"], "base27b_holistic": C["purple"],
         "llm_deepseek_decomposed": C["orange"], "llm_deepseek": C["orange"], "llm_qwen235b": C["orange"], "clf_src": C["gray"], "clf_src_cal": C["gray"],
         "string_match": C["green"], "jev_decomposed": C["blue"]}


def setup_fonts():
    for f in FONTS.glob("*.ttf"):
        fm.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "Nunito ExtraLight", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": C["gray"], "axes.labelcolor": "#222222", "xtick.color": "#333333",
        "ytick.color": "#222222", "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })
    return fm.FontProperties(fname=str(FONTS / "fredoka-600-normal.ttf"))


def compute() -> dict:
    data = {"diff": {}, "ours": {}, "risk": {}}
    for b in M.BENCH_NAMES:
        it, S = P.load(b)
        J = P.judged(it, S)
        T = P.test_half(it, J)
        data["ours"][b] = roc_auc_score([it[i]["label"] for i in T], [S["jev_decomposed"][i]["score"] for i in T])
        for g in M.MAIN_GUARDS[1:]:
            m, lo, hi, _ = E.boot_auc_diff_grouped(it, T, S["jev_decomposed"], S[g])
            data["diff"][f"{b}|{g}"] = [m, lo, hi]
        for g in M.RISK_GUARDS:
            r = E.grouped_risk(it, S, g, J, 0.05, 0.10, reps=200)
            data["risk"][f"{b}|{g}"] = {k: list(v) for k, v in r.items()}
        print("computed", b, flush=True)
    json.dump(data, open(CACHE, "w"), indent=1)
    return data


# 图里只画每个判官的主变体；单问题的开源变体（JEV-27B、两个读出）只在 Table 里
DIFF_GUARDS = [g for g in M.MAIN_GUARDS[1:] if g not in ("jev27b_holistic", "readout_holistic", "base27b_holistic")]
RISK_FIG_GUARDS = [g for g in M.RISK_GUARDS if g != "readout_decomposed"]


def fig_diff(d, title_font):
    guards = DIFF_GUARDS
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.35), sharey=True)
    vals = [d["diff"][f"{b}|{g}"] for b in M.BENCH_NAMES for g in guards]
    lo_all = min(v[1] for v in vals); hi_all = max(v[2] for v in vals)
    for ax, (b, name) in zip(axes, M.BENCH_NAMES.items()):
        ax.axvspan(lo_all - 0.05, 0, color=C["light"], zorder=0)
        ax.axvline(0, color=C["gray"], lw=0.7, zorder=1)
        for k, g in enumerate(guards):
            m, lo, hi = d["diff"][f"{b}|{g}"]
            y = len(guards) - 1 - k
            sig = lo > 0
            ax.plot([lo, hi], [y, y], color=GROUP[g], lw=1.3, solid_capstyle="round", zorder=2)
            ax.plot(m, y, "o", ms=4.2, mfc=GROUP[g] if sig else "white", mec=GROUP[g], mew=1.1, zorder=3)
        ax.set_xlim(lo_all - 0.04, hi_all + 0.03)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4]); ax.set_xticklabels(["0", ".1", ".2", ".3", ".4"])
        ax.set_title(f"{name}", fontproperties=title_font, fontsize=8.5, color="#222222", pad=5)
        ax.grid(axis="x", color="#dde1e6", lw=0.5, zorder=0)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(guards)))
    axes[0].set_yticklabels([SHORT[g] for g in reversed(guards)])
    for t, g in zip(axes[0].get_yticklabels(), reversed(guards)):
        t.set_color(GROUP[g])
    fig.text(0.58, 0.035, "AUROC of MinGuard minus AUROC of the baseline (right of 0: MinGuard better)",
             ha="center", fontsize=7)
    leg = [Line2D([], [], marker="o", ls="", mfc=C["gray"], mec=C["gray"], label="95% interval excludes 0"),
           Line2D([], [], marker="o", ls="", mfc="white", mec=C["gray"], label="includes 0")]
    fig.legend(handles=leg, loc="lower left", ncol=1, fontsize=6, bbox_to_anchor=(0.0, -0.005),
               handletextpad=0.3, labelspacing=0.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
    fig.savefig(OUT / "fig_diff.pdf")
    plt.close(fig)


def fig_risk(d, title_font):
    guards = RISK_FIG_GUARDS
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.2), sharey=True)
    for ax, (b, name) in zip(axes, M.BENCH_NAMES.items()):
        for k, g in enumerate(guards):
            r = d["risk"][f"{b}|{g}"]
            a, bl = r["allow"][0], r["block"][0]
            y = len(guards) - 1 - k
            h = 0.68
            ax.barh(y, a, height=h, color=C["green"], lw=0)
            ax.barh(y, bl, left=a, height=h, color=C["red"], lw=0)
            ax.barh(y, 1 - a - bl, left=a + bl, height=h, color="#F6D4B5", lw=0)
            if g == "jev_decomposed":
                ax.barh(y, 1, height=h + 0.12, fill=False, ec="#222222", lw=0.8)
            ax.text(1.04, y, f"{100 * r['miss'][0]:.1f}", va="center", ha="left", fontsize=6, color=C["gray"])
            ax.text(1.27, y, f"{100 * r['false_block'][0]:.1f}", va="center", ha="left", fontsize=6,
                    color=C["gray"])
        ax.text(1.04, len(guards) - 0.35, "miss", fontsize=6, color=C["gray"], ha="left", va="bottom")
        ax.text(1.27, len(guards) - 0.35, "f.b.", fontsize=6, color=C["gray"], ha="left", va="bottom")
        ax.set_xlim(0, 1.5)
        ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"], fontsize=6)
        ax.spines["bottom"].set_bounds(0, 1)
        ax.set_title(name, fontproperties=title_font, fontsize=8.5, color="#222222", pad=8)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(guards)))
    axes[0].set_yticklabels([SHORT[g] for g in reversed(guards)])
    for t, g in zip(axes[0].get_yticklabels(), reversed(guards)):
        t.set_color(GROUP[g])
        if g == "jev_decomposed":
            t.set_fontweight("bold")
    leg = [plt.Rectangle((0, 0), 1, 1, color=C["green"], label="allowed"),
           plt.Rectangle((0, 0), 1, 1, color=C["red"], label="blocked"),
           plt.Rectangle((0, 0), 1, 1, color="#F6D4B5", label="escalated"),
           Line2D([], [], ls="", label="numbers: mean miss and false-block rate, %")]
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=6, bbox_to_anchor=(0.55, -0.015),
               handlelength=1.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.4)
    fig.savefig(OUT / "fig_risk.pdf")
    plt.close(fig)


def fig_budget(title_font):
    res = json.load(open(P.RUNS / "label_budget.json"))
    ns = [25, 50, 100, 200]
    col = {"heldout": C["purple"], "real": C["blue"], "new_model": C["orange"], "new_db": C["green"]}
    fig, ax = plt.subplots(figsize=(3.5, 1.9))   # 与 \columnwidth（3.5in）同宽，印刷字号即设定字号
    for b, name in M.BENCH_NAMES.items():
        for g, ls, lw, mk in (("jev_decomposed", "-", 1.6, "o"), ("jev27b_decomposed", (0, (2.5, 1.5)), 1.0, None)):
            pts = [(n, res[f"{b}|{g}|{n}"]["auto"]) for n in ns if f"{b}|{g}|{n}" in res]
            x, y = zip(*pts)
            ax.plot(x, y, ls=ls, lw=lw, color=col[b], marker=mk, ms=3.5, mfc="white", mew=1.1,
                    label=name if g == "jev_decomposed" else None)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off()
    ax.set_ylim(0, 1.04); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("labeled questions used for calibration")
    ax.set_ylabel("judged requests decided")
    ax.grid(color="#dde1e6", lw=0.5)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=C["gray"], ls=(0, (2.5, 1.5)), lw=1.0)); l.append("JEV-27B")
    ax.legend(h, l, loc="lower right", fontsize=6, ncol=1, handlelength=1.8, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_budget.pdf")
    plt.close(fig)


def main() -> None:
    title_font = setup_fonts()
    d = compute() if "--recompute" in sys.argv or not CACHE.exists() else json.load(open(CACHE))
    fig_diff(d, title_font)
    fig_risk(d, title_font)
    fig_budget(title_font)
    p90m = max(v["miss"][1] for v in d["risk"].values()); p90f = max(v["false_block"][1] for v in d["risk"].values())
    print(f"max 90th-percentile miss {p90m:.3f}, false block {p90f:.3f}")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
