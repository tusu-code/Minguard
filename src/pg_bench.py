"""事实层在 PostgreSQL 上的开销，以及 EXPLAIN 估计行数能否代替试执行。

对自然分布的 3,893 条 OmniSQL-7B 请求：
  1. sqlglot 把 SQLite 方言转成 PostgreSQL，标识符统一小写加引号（与 pg_import.py 的建表一致）；
  2. 解析+血缘仍在 SQLite 的 schema 上做（与方言无关），这里只测数据库侧；
  3. 试执行：服务器端游标，最多取 1,001 行，statement_timeout 5 s，与 SQLite 版相同；
  4. 聚合查询再跑"最小分组人数"改写（sqlfacts.contributors 的同一改写）；
  5. EXPLAIN (FORMAT JSON) 取规划器估计的结果行数与耗时。
先把每张表扫一遍预热缓存。结果写 runs/pg_bench.json。
用法：FACTS_VERSION=1 python pg_bench.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import psycopg2
import sqlglot
from sqlglot import exp

import dbmeta
import guards as G
import sqlfacts

PG = dict(host="/tmp", port=5433, user="postgres")
CAP = sqlfacts.ROW_CAP + 1
TIMEOUT_MS = int(sqlfacts.EXEC_TIMEOUT_S * 1000)


def to_pg(sql: str) -> str:
    tree = sqlglot.parse_one(sql, read="sqlite")
    for node in tree.find_all(exp.Identifier):
        node.set("this", node.name.lower())
        node.set("quoted", True)
    return tree.sql(dialect="postgres")


def contributors_sql(db: str, sql: str) -> str | None:
    """sqlfacts.contributors 的改写，只返回 SQL 不执行。"""
    tree = sqlglot.parse_one(sql, read=sqlfacts.DIALECT)
    if not isinstance(tree, exp.Select):
        return None
    frm = tree.args.get("from") or tree.args.get("from_")
    nodes = ([frm.this] if frm is not None else []) + [j.this for j in tree.args.get("joins") or []]
    key = None
    for n in nodes:
        if isinstance(n, exp.Table):
            t = next((k for k in sqlfacts.PERSON_KEY if k.lower() == n.name.lower()), None)
            if t:
                key = f'{n.alias_or_name}."{sqlfacts.PERSON_KEY[t]}"'
                break
    if key is None:
        return None
    new = tree.copy()
    new.set("expressions", [sqlglot.parse_one(f"COUNT(DISTINCT {key})", read=sqlfacts.DIALECT).as_("n")])
    for k in ("order", "limit", "offset", "having", "distinct"):
        new.set(k, None)
    return f"SELECT MIN(n) FROM ({new.sql(dialect=sqlfacts.DIALECT)}) AS sub"


def warm(con) -> None:
    cur = con.cursor()
    cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    for (t,) in cur.fetchall():
        cur.execute(f'SELECT count(*) FROM "{t}"')
    con.commit()


def timed(con, sql: str, fetch_cap: bool):
    """返回 (毫秒, 行数或首值, 错误)。"""
    cur = con.cursor(name="c") if fetch_cap else con.cursor()
    t0 = time.perf_counter()
    try:
        cur.execute(sql)
        if fetch_cap:
            out = len(cur.fetchmany(CAP))
        else:
            out = cur.fetchone()
        err = None
    except psycopg2.Error as e:
        out, err = None, type(e).__name__ + ": " + str(e).splitlines()[0][:120]
    ms = (time.perf_counter() - t0) * 1000
    try:
        cur.close()
    except psycopg2.Error:
        pass
    con.rollback()
    return ms, out, err


def main() -> None:
    items = [json.loads(l) for l in open(dbmeta.ROOT / "data/natural_omnisql.jsonl")]
    cons = {}
    for db in {x["db"] for x in items}:
        c = psycopg2.connect(dbname=db.lower(), **PG)
        c.cursor().execute(f"SET statement_timeout = {TIMEOUT_MS}")
        c.commit()
        warm(c)
        cons[db] = c
    recs = []
    for k, x in enumerate(items):
        db, sql = x["db"], x["sql"]
        f = sqlfacts.analyze(db, sql)
        r = {"id": x.get("id"), "db": db}
        if f.statement != "select" or f.parse_error:
            r["skip"] = "not a parsable select"
            recs.append(r)
            continue
        con = cons[db]
        try:
            pg_sql = to_pg(sql)
        except Exception as e:  # noqa: BLE001
            r["skip"] = "translate: " + type(e).__name__
            recs.append(r)
            continue
        sq_rows, sq_f = sqlfacts.dry_run(db, sql, sqlfacts.Facts(statement="select"))
        r["sqlite_ms"] = sq_f.exec_ms
        r["sqlite_rows"] = None if sq_rows is None else len(sq_rows)
        r["exec_ms"], r["rows"], r["err"] = timed(con, pg_sql, True)
        t0 = time.perf_counter()
        ms, plan, perr = timed(con, "EXPLAIN (FORMAT JSON) " + pg_sql, False)
        r["explain_ms"] = ms
        r["est_rows"] = None if perr or plan is None else float(plan[0][0]["Plan"]["Plan Rows"])
        r["contrib_ms"] = None
        if f.aggregate_query and r["err"] is None:
            cs = contributors_sql(db, sql)
            if cs:
                try:
                    ms, val, cerr = timed(con, to_pg(cs), False)
                    r["contrib_ms"], r["contrib_err"] = ms, cerr
                except Exception:  # noqa: BLE001
                    r["contrib_err"] = "translate"
        r["individual"] = f.individual_level(db)
        r["short_allow"] = G.short_circuit(x, sqlfacts.facts_for(db, sql, execute=False)) is not None
        recs.append(r)
        if k % 500 == 0:
            print(k, flush=True)
    json.dump(recs, open(dbmeta.ROOT / "runs/pg_bench.json", "w"))
    print("done", len(recs))


if __name__ == "__main__":
    main()
