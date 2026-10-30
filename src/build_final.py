"""最终测试集：v2 冻结后新生成的 SQL（新模型 / 新库），自动打标签 → 严格子集规则 → 分层抽样。

分两块报告：
  new_db    OmniSQL-7B 与 Qwen2.5-Coder-7B 在 3 个新库上的输出（库、题目都没在设计中出现过）
  new_model Qwen2.5-Coder-7B 在原 4 库上的输出（题目见过，生成模型是新的）
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import Counter

import dbmeta

ROOT = dbmeta.ROOT
NEG_CAP = {"real:subset_of_gold": 250, "real:no_personal_output": 120, "realR:allowed_columns": 120,
           "realR:restricted_column": 120}


def main() -> None:
    gens = {"new_db": ["runs/gen_coder7b_new.jsonl", "runs/gen_omnisql7b_new.jsonl"],
            "new_model": ["runs/gen_coder7b_pilotdb.jsonl"]}
    rng = random.Random(20261025)
    final = []
    for block, files in gens.items():
        raw = ROOT / f"data/final_{block}_raw.jsonl"
        subprocess.run([sys.executable, "label_real.py", *[str(ROOT / f) for f in files], "--out", str(raw),
                        "--workers", str(min(64, __import__("os").cpu_count()))], check=True)
        strict = ROOT / f"data/final_{block}_strict.jsonl"
        subprocess.run([sys.executable, "strict_real.py", str(raw), "-", str(strict)], check=True)
        xs = [json.loads(l) for l in open(strict) if not json.loads(l)["category"].startswith("real:non-select")]
        by: dict = {}
        for x in xs:
            by.setdefault(x["category"], []).append(x)
        for cat, v in by.items():
            n = NEG_CAP.get(cat, len(v))
            pick = v if n >= len(v) else rng.sample(v, n)
            for x in pick:
                x["block"] = block
                x["id"] = f"{block}:{x['id']}"
            final += pick
    out = ROOT / "data/final_bench.jsonl"
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in final))
    print(f"最终测试集 {len(final)} 条 → {out}")
    print(Counter((x["block"], x["policy"], x["label"]) for x in final))
    print(Counter((x["block"], x["category"]) for x in final))


if __name__ == "__main__":
    main()
