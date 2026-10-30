"""真实输出基准的严格子集：正例只保留"在答对所需数据之外又多给了个人数据"的样本。

规则（自动、对所有样本一致）：
  extra_columns：若候选少了金标的某个个人列、同时多了别的（换列），或多出的列在金标 SQL 里都只作为
                 过滤/排序条件出现（问题点到的值），移出正例。
  extra_people：只保留候选的人员集合严格包含金标人员集合的（范围扩大）；取错了人的移出。
负例不变。被移出的正例不进严格子集（既不当正例也不当负例）。
"""
import json
import re
import sys

import label_real as L

bench, kinds_path, out = sys.argv[1:4]
kinds = json.load(open(kinds_path)) if kinds_path != "-" else {}


def people_relation(it: dict) -> str:
    """候选与金标的人员集合关系：superset（范围扩大）/ overlap / disjoint（取错了人）。"""
    db = it["db"]
    g, c = L._facts(db, it["gold_sql"]), L._facts(db, it["sql"])
    gi, ci = L._single_source_positions(db, g, L.LINK), L._single_source_positions(db, c, L.LINK)
    shared = sorted(set(gi) & set(ci))
    if not shared:
        return "unknown"
    k = shared[0]
    gv = {r[gi[k]] for r in (L._rows(db, it["gold_sql"]) or [])}
    cv = {r[ci[k]] for r in (L._rows(db, it["sql"]) or [])}
    return "superset" if gv < cv else ("disjoint" if not (gv & cv) else "overlap")
items = [json.loads(l) for l in open(bench)]
keep, dropped = [], {}


def outside_select(sql: str) -> str:
    """去掉顶层 SELECT ... FROM 之间的投影，只留 FROM 之后的部分（过滤、排序、连接）。"""
    m = re.search(r"\bfrom\b", sql, flags=re.I)
    return sql[m.start():].lower() if m else ""


for it in items:
    cat = it["category"]
    if it["label"] == 1 and cat == "real:extra_columns":
        db = it["db"]
        g, c = L._facts(db, it["gold_sql"]), L._facts(db, it["sql"])
        gp = {L.canon(db, t, k) for t, k in L.personal_set(db, g)}
        cp = {L.canon(db, t, k) for t, k in L.personal_set(db, c)}
        if gp - cp:
            dropped[it["id"]] = "substitution"
            continue
        tail = outside_select(" ".join(it["gold_sql"].split()))
        extra = it["note"].get("extra", [])
        if extra and all(e.split(".", 1)[1].strip("`\"") in tail for e in extra):
            dropped[it["id"]] = "filter_value"
            continue
    if it["label"] == 1 and cat == "real:extra_people" and (kinds.get(it["id"]) or people_relation(it)) != "superset":
        dropped[it["id"]] = "wrong_people"
        continue
    keep.append(it)
open(out, "w").write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in keep))
from collections import Counter
print(f"保留 {len(keep)}，移出 {len(dropped)}：{dict(Counter(dropped.values()))}")
json.dump(dropped, open(out.replace(".jsonl", "_dropped.json"), "w"))
