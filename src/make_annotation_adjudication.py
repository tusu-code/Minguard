"""第三位标注者盲标仲裁：New DBs 测试半边里不在两人共识中的 74 条（判定不一致 55 条，含歧义类型 19 条）。

第三位标注者是另一位非作者同事，看不到自动标签、守卫分数和前两人的判定，格式与前几次相同。
多数规则（在拿到第三份标注之前定下）：每人的标注映射为 1（四种越量类型之一）、0（不越量）或
歧义（歧义类型或无法判断）；三人中至少两人给出同一个值即取该值，取到歧义或三人各不相同则排除。
两人共识的 137 条不变，最终标签 = 共识 137 条 + 仲裁后得到 0/1 的样本。
输出：
  annotation/新数据库仲裁_74条_标注人3.xlsx
  annotation/answer_key_adjudication.json   只留本地
用法：FACTS_VERSION=1 python make_annotation_adjudication.py
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

import dbmeta
import make_annotation as MA
import make_tables as M
import paper_data as P

ROOT = dbmeta.ROOT


def lab(v, t):
    return 0 if v == "不越量" else (1 if t in M.HUMAN_CORE else None)


def newdb_test_records() -> list[dict]:
    it, S = P.load("new_db")
    te = set(P.test_half(it, P.judged(it, S)))
    recs = [r for r in json.load(open(ROOT / "annotation/human_labels_newdb.json")) if r["id"] in te]
    return recs + json.load(open(ROOT / "annotation/human_labels_newdb_test.json"))


def majority(*ys):
    c = Counter("amb" if y is None else y for y in ys)
    v, n = c.most_common(1)[0]
    return v if n >= 2 and v != "amb" else None


def main() -> None:
    recs = newdb_test_records()
    todo = [r for r in recs if not (lab(r["verdict1"], r["type1"]) is not None
                                    and lab(r["verdict1"], r["type1"]) == lab(r["verdict2"], r["type2"]))]
    raw = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/final_new_db_raw.jsonl")}
    it, _ = P.load("new_db")
    xs = []
    for r in todo:
        x = dict(raw[r["id"].split(":", 1)[1]])
        x["bench_id"], x["in_eval_subset"] = r["id"], True
        x["label"], x["category"] = it[r["id"]]["label"], it[r["id"]]["category"]
        xs.append(x)
    rng = random.Random(20261107)
    by_q = defaultdict(list)
    for x in xs:
        by_q[(x["db"], x["qid"])].append(x)
    qs = list(by_q)
    rng.shuffle(qs)
    picked = []
    for q in qs:
        rng.shuffle(by_q[q])
        picked += by_q[q]
    print("仲裁", len(picked), "题数", len(qs), Counter(x["category"] for x in picked))
    wb, _ = MA.write_xlsx(picked, None, dbs=dbmeta.HELDOUT_DBS)
    key = MA.write_annotation(wb, picked, ROOT / "annotation/新数据库仲裁_74条_标注人3.xlsx")
    for k, x in zip(key, picked):
        k["id"] = x["bench_id"]
    json.dump(key, open(ROOT / "annotation/answer_key_adjudication.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
