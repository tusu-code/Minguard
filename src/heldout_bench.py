"""留出基准：试点没见过的攻击写法（H1–H6）、良性改写（N1–N2）、测试时才出现的新策略（alt）。

标签不靠构造方式直接给，统一用 label_real.label_pair（以金标输出为"所需"参照）判定，
攻击类只保留判为 1 的，良性类只保留判为 0 的，避免"变异没真正越量"混进来。
alt 策略：金标输出的个人列含 alt 受限列 → 1；全部在 alt 允许清单内 → 0；其余不进基准。
"""

from __future__ import annotations

import json
import random
from collections import Counter

import dbmeta
import heldout_mut as H
import label_real as L
import sqlfacts

OUT = dbmeta.ROOT / "data/heldout_bench.jsonl"
SEED = 20261024
CAP = 15


def main() -> None:
    rng = random.Random(SEED)
    data = [x for x in json.load(open(dbmeta.BIRD_DEV)) if x["db_id"] in dbmeta.PILOT_DBS]
    rng.shuffle(data)
    items, count, drop = [], Counter(), Counter()

    def add(cat, policy, label, x, sql, note=None):
        if count[(x["db_id"], cat)] >= CAP:
            return
        count[(x["db_id"], cat)] += 1
        items.append({"id": f"{x['question_id']}_{cat}", "qid": x["question_id"], "db": x["db_id"],
                      "category": cat, "policy": policy, "label": label, "question": x["question"],
                      "evidence": x["evidence"], "sql": sql, "gold_sql": x["SQL"], "note": note,
                      "split": "cal" if x["question_id"] % 2 == 0 else "test"})

    for x in data:
        db, sql = x["db_id"], x["SQL"]
        g = sqlfacts.facts_for(db, sql)
        if g.exec_error or g.parse_error or not g.result_rows or g.individual_level(db) is None:
            continue
        for cat, fn in {**H.ATTACKS, **H.BENIGN}.items():
            if count[(db, cat)] >= CAP:
                continue
            try:
                new = fn(db, sql, rng) if fn.__code__.co_argcount == 3 else fn(db, sql)
            except Exception:  # noqa: BLE001
                new = None
            if not new:
                continue
            lab = L.label_pair(db, sql, new)
            want = 1 if cat.startswith("H") else 0
            if lab is None or lab["label"] != want:
                drop[(cat, "label_mismatch" if lab else "unlabelable")] += 1
                continue
            add(cat, "default", want, x, new, note=lab)

        # 新策略：同一条金标在 alt 策略下的标签
        if g.individual_level(db):
            po = g.personal_out(db)
            if any(dbmeta.restricted_under("alt", db, t, c) for t, c, _ in po):
                add("P1", "alt", 1, x, sql, note={"restricted": [f"{t}.{c}" for t, c, _ in po
                                                             if dbmeta.restricted_under("alt", db, t, c)]})
            elif all(dbmeta.allowed_under("alt", db, t, c) for t, c, _ in po):
                add("P0", "alt", 0, x, sql)

    OUT.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    print(f"写出 {len(items)} 条 → {OUT}")
    for k in sorted(Counter((it["category"], it["label"]) for it in items).items()):
        print(" ", k)
    print("丢弃:", dict(drop))


if __name__ == "__main__":
    main()
