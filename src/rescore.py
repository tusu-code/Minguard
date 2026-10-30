"""用已存的分解问题概率按新融合规则重算分数（不重新调用判官）。"""
import json
import sys

import guards as G
import sqlfacts

bench, src, dst = sys.argv[1:4]
items = {json.loads(l)["id"]: json.loads(l) for l in open(bench)}
cache = {}
n_changed = 0
with open(dst, "w") as fo:
    for line in open(src):
        r = json.loads(line)
        if r["id"] not in items:          # 基准里已剔除的样本
            continue
        if r["guard"] in ("jev_decomposed", "readout_decomposed") and r.get("probs") and not r.get("det"):
            it = items[r["id"]]
            if it["id"] not in cache:
                cache[it["id"]] = bool(sqlfacts.facts_for(it["db"], it["sql"]).individual_level(it["db"]))
            s, why = G.fuse(r["probs"], individual=cache[it["id"]])
            if abs(s - r["score"]) > 1e-9:
                n_changed += 1
            r["score"], r["why"] = s, why
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{bench}: 改动 {n_changed} 条 → {dst}")
