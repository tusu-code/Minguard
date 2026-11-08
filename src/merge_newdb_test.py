"""合并 New DBs 测试半边 165 条补标（两位标注人各一份）为 annotation/human_labels_newdb_test.json。
用法：python merge_newdb_test.py ../annotation/sheet_newdb_test_165_annotator1.xlsx ../annotation/sheet_newdb_test_165_annotator2.xlsx
"""
import json
import sys

from openpyxl import load_workbook

import dbmeta

# 发布包不含答案键：样本编号与 id 的对应取自已合并的 human_labels_newdb_test.json
key = json.load(open(dbmeta.ROOT / "annotation/human_labels_newdb_test.json"))
sheets = [list(load_workbook(p, data_only=True)["标注"].iter_rows(values_only=True))[1:] for p in sys.argv[1:3]]
out = []
for k, r1, r2 in zip(key, *sheets):
    assert k["no"] == r1[0] == r2[0]
    out.append({"no": k["no"], "id": k["id"], "auto_label": k["auto_label"], "auto_category": k["auto_category"],
                "in_eval": True, "verdict1": r1[10], "type1": r1[11], "correct1": r1[12], "note1": r1[13] or "",
                "verdict2": r2[10], "type2": r2[11], "correct2": r2[12], "note2": r2[13] or ""})
json.dump(out, open(dbmeta.ROOT / "annotation/human_labels_newdb_test.json", "w"), ensure_ascii=False, indent=1)
print("合并", len(out))
