"""审稿意见补救：在三个新数据库上抽 150 条做盲标，格式与 200 条那份相同（说明、个人数据列、标注三页）。

优先抽评测子集（final_bench.jsonl 里 new_db 块）中的样本，这样每条都有全部守卫的打分；每题尽量只取一条。
输出：
  annotation/新数据库人工标注_150条_标注人A.xlsx / _标注人B.xlsx（内容相同，两人各自独立标）
  annotation/answer_key_newdb.json   只留本地
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter

import dbmeta
import make_annotation as MA

ROOT = dbmeta.ROOT
QUOTA = {"real:extra_columns": 35, "real:extra_people": 30, "real:aggregate->rows": 25,
         "real:subset_of_gold": 40, "real:echo_named": 5, "real:no_personal_output": 15}


def sample() -> list[dict]:
    rng = random.Random(20261026)
    xs = [json.loads(l) for l in open(ROOT / "data/final_new_db_raw.jsonl")]
    # final_bench 里的编号带 "new_db:" 前缀
    in_eval = {json.loads(l)["id"].split(":", 1)[1] for l in open(ROOT / "data/final_bench.jsonl")
               if json.loads(l).get("block") == "new_db"}
    picked, used_q = [], set()
    for cat, n in QUOTA.items():
        pool = [x for x in xs if x["category"] == cat]
        rng.shuffle(pool)
        pool.sort(key=lambda x: (x["id"] not in in_eval, x["qid"] in used_q))
        chosen, qs = [], set()
        for x in pool:
            if len(chosen) >= n:
                break
            if x["qid"] not in qs:
                chosen.append(x)
                qs.add(x["qid"])
        chosen += [x for x in pool if x not in chosen][: n - len(chosen)]
        used_q |= qs
        for c in chosen:
            c["in_eval_subset"] = c["id"] in in_eval
        picked += chosen
    rng.shuffle(picked)
    print("抽样", len(picked), Counter(x["category"] for x in picked),
          "在评测子集内", sum(x["in_eval_subset"] for x in picked), "题数", len({x["qid"] for x in picked}))
    return picked


def main() -> None:
    out = ROOT / "annotation"
    picked = sample()
    wb, _ = MA.write_xlsx(picked, None, dbs=dbmeta.HELDOUT_DBS)
    a = out / "新数据库人工标注_150条_标注人A.xlsx"
    key = MA.write_annotation(wb, picked, a)
    shutil.copy(a, out / "新数据库人工标注_150条_标注人B.xlsx")
    json.dump(key, open(out / "answer_key_newdb.json", "w"), ensure_ascii=False, indent=1)
    print("写出", a, "及标注人 B 副本")


if __name__ == "__main__":
    main()
