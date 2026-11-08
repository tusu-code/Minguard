"""合并第三位标注者的仲裁标注（74 条），按事先定好的多数规则得到 New DBs 测试半边的最终人工标签。
输出 annotation/human_labels_newdb_adjudicated.json：每条请求带三人的判定与最终标签（1/0/None=排除）。
用法：python merge_adjudication.py ../annotation/sheet_newdb_adjudication_74_annotator3.xlsx（需本地答案键 answer_key_adjudication.json；发布包已含合并结果）
"""
import json
import sys

from openpyxl import load_workbook

import dbmeta
import make_annotation_adjudication as A

key = json.load(open(dbmeta.ROOT / "annotation/answer_key_adjudication.json"))
rows = list(load_workbook(sys.argv[1], data_only=True)["标注"].iter_rows(values_only=True))[1:]
third = {}
for k, r in zip(key, rows):
    assert k["no"] == r[0]
    third[k["id"]] = {"verdict3": r[10], "type3": r[11], "correct3": r[12], "note3": r[13] or ""}
out = []
for r in A.newdb_test_records():
    y1, y2 = A.lab(r["verdict1"], r["type1"]), A.lab(r["verdict2"], r["type2"])
    rec = {k: r[k] for k in ("id", "auto_label", "auto_category", "verdict1", "type1", "verdict2", "type2")}
    if y1 is not None and y1 == y2:
        rec["final"], rec["source"] = y1, "consensus"
    else:
        t = third[r["id"]]
        rec.update(t)
        rec["final"], rec["source"] = A.majority(y1, y2, A.lab(t["verdict3"], t["type3"])), "adjudicated"
    out.append(rec)
json.dump(out, open(dbmeta.ROOT / "annotation/human_labels_newdb_adjudicated.json", "w"), ensure_ascii=False, indent=1)
print("写出", len(out))
