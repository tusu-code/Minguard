"""汇总 pg_bench.py 的结果（runs/pg_bench.json → runs/pg_summary.json），论文 RQ4 的 PostgreSQL 数字都出自这里。"""

from __future__ import annotations

import collections
import json

import numpy as np

import dbmeta


def pct(v):
    v = np.asarray(v, dtype=float)
    return {p: float(np.percentile(v, p)) for p in (50, 95, 99)} | {"max": float(v.max()), "n": len(v)}


def main() -> None:
    R = json.load(open(dbmeta.ROOT / "runs/pg_bench.json"))
    ok = [r for r in R if "skip" not in r]
    err = [r for r in ok if r["err"]]
    both = [r for r in ok if not r["err"] and r["sqlite_rows"] is not None]
    est = np.array([min(r["est_rows"], 1001) for r in both])
    act = np.array([r["rows"] for r in both])
    ratio = np.maximum(est, 1) / np.maximum(act, 1)
    ratio = np.maximum(ratio, 1 / ratio)
    empty = [r for r in both if r["rows"] == 0]
    out = {
        "n": len(R), "pg_errors": len(err),
        "pg_error_kinds": collections.Counter(r["err"].split(":")[0] for r in err).most_common(),
        "timeouts": sum("QueryCanceled" in r["err"] for r in err),
        "both_ran": len(both),
        "rows_equal": sum(r["rows"] == r["sqlite_rows"] for r in both),
        "pg_exec": pct([r["exec_ms"] for r in both]),
        "sqlite_exec": pct([r["sqlite_ms"] for r in both]),
        "pg_contrib": pct([r["contrib_ms"] for r in ok if r.get("contrib_ms") is not None]),
        "pg_explain": pct([r["explain_ms"] for r in ok if r["est_rows"] is not None]),
        "explain_within_2x": float((ratio <= 2).mean()),
        "explain_within_10x": float((ratio <= 10).mean()),
        "empty_results": len(empty),
        "empty_estimated_nonzero": sum(r["est_rows"] >= 1 for r in empty),
    }
    json.dump(out, open(dbmeta.ROOT / "runs/pg_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
