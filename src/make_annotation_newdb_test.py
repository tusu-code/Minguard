"""补标：New DBs 测试半边里还没有人工标签的 165 条判官请求，两位标注人各自盲标。

格式与前两次相同（说明、个人数据列、标注三页），表里不出现自动标签和守卫分数。
同一道题的候选排在一起（题目顺序随机、题内顺序随机），省去重复读题。
输出：
  annotation/新数据库测试集补标_165条_标注人1.xlsx / _标注人2.xlsx（内容相同）
  annotation/answer_key_newdb_test.json   只留本地
用法：FACTS_VERSION=1 python make_annotation_newdb_test.py
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict

import dbmeta
import make_annotation as MA
import paper_data as P

ROOT = dbmeta.ROOT


def todo() -> list[dict]:
    it, S = P.load("new_db")
    te = P.test_half(it, P.judged(it, S))
    H = json.load(open(ROOT / "annotation/human_labels_newdb.json"))
    done = {r["id"] for r in (H if isinstance(H, list) else H.values())}
    raw = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/final_new_db_raw.jsonl")}
    xs = []
    for i in te:
        if i in done:
            continue
        x = dict(raw[i.split(":", 1)[1]])
        x["bench_id"], x["in_eval_subset"] = i, True
        x["label"], x["category"] = it[i]["label"], it[i]["category"]   # 评测用的（严格）标签，只进答案键
        xs.append(x)
    rng = random.Random(20261106)
    by_q = defaultdict(list)
    for x in xs:
        by_q[(x["db"], x["qid"])].append(x)
    qs = list(by_q)
    rng.shuffle(qs)
    out = []
    for q in qs:
        g = by_q[q]
        rng.shuffle(g)
        out += g
    print("待标", len(out), "题数", len(qs), Counter(x["category"] for x in out))
    return out


def main() -> None:
    picked = todo()
    wb, _ = MA.write_xlsx(picked, None, dbs=dbmeta.HELDOUT_DBS)
    a = ROOT / "annotation/新数据库测试集补标_165条_标注人1.xlsx"
    key = MA.write_annotation(wb, picked, a)
    for k, x in zip(key, picked):
        k["id"] = x["bench_id"]
    shutil.copy(a, ROOT / "annotation/新数据库测试集补标_165条_标注人2.xlsx")
    json.dump(key, open(ROOT / "annotation/answer_key_newdb_test.json", "w"), ensure_ascii=False, indent=1)
    print("写出", a, "及标注人 2 副本")


if __name__ == "__main__":
    main()
