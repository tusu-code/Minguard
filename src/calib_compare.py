"""同口径对比：在与 Fig. 4 相同的 200 次随机按题划分上，比较"全部候选进校准"和"按题分组校准"的漏放、误拦均值与 90 分位（回应审稿意见）。
用法：FACTS_VERSION=1 python calib_compare.py
"""
import numpy as np, collections
import paper_data as P, evaluate as E
def run(it,S,g,ids,grouped,reps=200,a=.05,b=.10):
    rng=np.random.default_rng(0); byq=collections.defaultdict(list)
    for i in ids: byq[(it[i]["db"],it[i]["qid"],it[i]["policy"])].append(i)
    qs=list(byq); R=collections.defaultdict(list)
    for _ in range(reps):
        p=rng.permutation(len(qs)); cq=[qs[k] for k in p[:len(qs)//2]]; tq=[qs[k] for k in p[len(qs)//2:]]
        if grouped: cal=[byq[q][rng.integers(len(byq[q]))] for q in cq]
        else: cal=[i for q in cq for i in byq[q]]
        l,h=E.thresholds([it[i]["label"] for i in cal],[S[g][i]["score"] for i in cal],a,b)
        W=wp=wn=mp=fb=au=0.0
        for q in tq:
            w=1/len(byq[q])
            for i in byq[q]:
                s,y=S[g][i]["score"],it[i]["label"]; W+=w; au+=w*(s<l or s>=h)
                if y: wp+=w; mp+=w*(s<l)
                else: wn+=w; fb+=w*(s>=h)
        R["auto"].append(au/W); R["miss"].append(mp/max(wp,1e-9)); R["fb"].append(fb/max(wn,1e-9))
    return {k:(np.mean(v),np.percentile(v,90)) for k,v in R.items()}, np.mean(np.array(R["fb"])>b), np.mean(np.array(R["miss"])>a)
for b in ["real","new_model","heldout","new_db"]:
    it,S=P.load(b); J=P.judged(it,S)
    for grp in (False,True):
        r,ofb,omiss=run(it,S,"jev_decomposed",J,grp)
        print(f"{b:9s} {'grouped  ' if grp else 'all-cands'} auto {r['auto'][0]:.3f} miss mean {r['miss'][0]:.3f} p90 {r['miss'][1]:.3f} | fb mean {r['fb'][0]:.3f} p90 {r['fb'][1]:.3f} | splits over fb {ofb:.2f} over miss {omiss:.2f}")
