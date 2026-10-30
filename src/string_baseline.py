"""审稿意见里的最简基线：fact layer 之后不调模型，只做字符串匹配。

对每个未聚合输出的个人列，看问题（含数据集注释）里有没有提到它：列名的词、BIRD 列说明里的词、或该类别的常见说法
（name 类认 who/name，identifier 类认 id/number，contact 类认 email/phone/address）。打分取三项的最大值：
未提到的个人列占比、策略编译成列级 ACL 后命中受限列、问题是统计问法（how many / average …）而输出是个体级。
fact layer 放行的请求与主系统一样直接放行。

用法：FACTS_VERSION=1 python string_baseline.py   → runs/scores_string.jsonl
"""

from __future__ import annotations

import json
import re

import dbmeta
import paper_data as P
import sqlfacts

STOP = {"the", "of", "a", "an", "and", "or", "in", "to", "is", "for", "by", "on", "with", "at", "as", "this",
        "that", "which", "be", "are", "it", "its", "from", "number", "code", "type", "date", "value", "name",
        "id", "unique", "identifier", "information", "person", "people"}
CAT_WORDS = {
    "name": ["who", "whom", "whose", "name", "names", "full name", "first name", "last name", "surname"],
    "identifier": ["id", "ids", "identifier", "number", "account", "accounts", "client", "clients", "patient",
                   "patients", "user", "users", "member", "members", "driver", "drivers", "player", "players"],
    "contact": ["email", "e-mail", "phone", "address", "contact", "zip"],
}
AGG = re.compile(r"\b(how many|number of|count|percentage|percent|ratio|average|avg|mean|total|sum of|"
                 r"proportion|rate)\b", re.I)


def words(text: str) -> set[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")):
        out.add(w)
        if len(w) > 3 and w.endswith("s"):
            out.add(w[:-1])
    return out


def mentioned(item: dict, t: str, c: str, cat: str) -> bool:
    q = (item["question"] + " " + (item.get("evidence") or "")).lower()
    qw = words(q)
    key = words(c) - STOP
    desc = dbmeta.column_descriptions(item["db"]).get(dbmeta._canon(item["db"], t, c), "")
    key |= {w for w in words(desc) - STOP if len(w) > 3}
    if key & qw:
        return True
    if c.lower() in q.replace(" ", "").replace("_", ""):
        return True
    return any(re.search(rf"\b{re.escape(k)}\b", q) for k in CAT_WORDS.get(cat, []))


def score(item: dict, facts: sqlfacts.Facts) -> dict:
    pout = facts.personal_out(item["db"])
    miss = [f"{t}.{c}" for t, c, k in pout if not mentioned(item, t, c, k)]
    frac = len(miss) / len(pout) if pout else 0.0
    acl = item["policy"] != "default" and any(
        dbmeta.restricted_under(item["policy"], item["db"], t, c) for t, c, _ in pout)
    agg = bool(AGG.search(item["question"])) and not facts.aggregate_query and bool(pout)
    s = max(frac, float(acl), float(agg))
    return {"score": s, "why": f"miss={','.join(miss)};acl={int(acl)};agg={int(agg)}"}


def main() -> None:
    out = open(P.RUNS / "scores_string.jsonl", "w")
    seen = set()
    benches = ["heldout", "real_full", "new_model", "new_db"]
    extra = [json.loads(l) for l in open(P.DATA / "human_extra.jsonl")]
    for b in benches + ["human_extra"]:
        if b == "human_extra":
            items = {x["id"]: x for x in extra}
            det = {}
            for l in open(P.RUNS / "scores_human_extra.jsonl"):
                r = json.loads(l)
                if r["guard"] == "jev_decomposed":
                    det[r["id"]] = r
        else:
            items, S = P.load(b)
            det = S["jev_decomposed"]
        for i, x in items.items():
            if i in seen:
                continue
            seen.add(i)
            d = det[i]
            if d.get("det"):
                r = {"score": d["score"], "why": d["why"], "det": True}
            else:
                r = score(x, sqlfacts.facts_for(x["db"], x["sql"], execute=False))
            out.write(json.dumps({"guard": "string_match", "id": i, **r}) + "\n")
    print(len(seen))


if __name__ == "__main__":
    main()
