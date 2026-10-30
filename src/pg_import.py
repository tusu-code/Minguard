"""把 7 个 BIRD SQLite 库导入本机 PostgreSQL 16（端口 5433），用于测事实层在服务器数据库上的开销。

SQLite 是动态类型：按列的实际取值推断类型（全是整数 → bigint，全是数 → double precision，其余 text）。
表名、列名统一转成小写并加引号，查询改写时同样转小写（pg_bench.py），这样 SQLite 不区分大小写的写法在
PostgreSQL 里也能对上。导入后建与 SQLite 相同的主键，并 ANALYZE，让 EXPLAIN 有统计信息可用。
用法：python pg_import.py
"""

from __future__ import annotations

import io
import sqlite3
import time

import psycopg2

import dbmeta

PG = dict(host="/tmp", port=5433, user="postgres")
DBS = dbmeta.PILOT_DBS + dbmeta.HELDOUT_DBS


def q(name: str) -> str:
    return '"' + name.lower().replace('"', '""') + '"'


def infer(con, table: str, col: str) -> str:
    kinds = {r[0] for r in con.execute(
        f'SELECT DISTINCT typeof("{col}") FROM "{table}" WHERE "{col}" IS NOT NULL')}
    if not kinds or kinds == {"integer"}:
        return "bigint" if kinds else "text"
    if kinds <= {"integer", "real"}:
        return "double precision"
    return "text"


def fmt(v) -> str:
    if v is None:
        return r"\N"
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    s = str(v)
    return s.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def main() -> None:
    admin = psycopg2.connect(dbname="postgres", **PG)
    admin.autocommit = True
    for db in DBS:
        t0 = time.time()
        with admin.cursor() as c:
            c.execute(f"DROP DATABASE IF EXISTS {q(db)}")
            c.execute(f"CREATE DATABASE {q(db)}")
        src = sqlite3.connect(f"file:{dbmeta.db_path(db)}?mode=ro", uri=True)
        src.text_factory = lambda b: b.decode("utf-8", errors="replace")
        pg = psycopg2.connect(dbname=db.lower(), **PG)
        cur = pg.cursor()
        nrows = 0
        for (table,) in src.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
            info = list(src.execute(f'PRAGMA table_info("{table}")'))
            cols = [r[1] for r in info]
            types = [infer(src, table, c) for c in cols]
            pk = [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5] > 0]
            cur.execute(f"CREATE TABLE {q(table)} (" + ", ".join(f"{q(c)} {t}" for c, t in zip(cols, types)) + ")")
            buf = io.StringIO()
            for row in src.execute(f'SELECT * FROM "{table}"'):
                buf.write("\t".join(fmt(v) for v in row) + "\n")
                nrows += 1
            buf.seek(0)
            cur.copy_expert(f"COPY {q(table)} FROM STDIN", buf)
            if pk:
                try:
                    cur.execute(f"ALTER TABLE {q(table)} ADD PRIMARY KEY (" + ", ".join(q(c) for c in pk) + ")")
                except psycopg2.Error as e:   # BIRD 有少数主键重复，跳过即可
                    pg.rollback()
                    print("  pk skipped", table, str(e).splitlines()[0])
                    continue
            pg.commit()
        pg.commit()
        pg.autocommit = True
        pg.cursor().execute("ANALYZE")
        pg.close()
        print(f"{db}: {nrows} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
