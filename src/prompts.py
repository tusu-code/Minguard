"""Text-to-SQL 生成用的提示词：OmniSQL 官方格式（含"只输出所需列"指令）与朴素代理格式（不含）。"""

from __future__ import annotations

import sqlite3
from functools import lru_cache

import dbmeta


def _quote(name: str) -> str:
    return f'"{name}"' if not name.isidentifier() else name


@lru_cache(maxsize=None)
def schema_ddl(db: str, n_examples: int = 3) -> str:
    """CREATE TABLE 语句，列后附示例值注释（OmniSQL 的 db_details 格式）。"""
    con = sqlite3.connect(f"file:{dbmeta.db_path(db)}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    parts = []
    for table, cols in dbmeta.load_schema(db).items():
        info = con.execute(f'PRAGMA table_info("{table}")').fetchall()
        pks = [r[1] for r in info if r[5]]
        fks = con.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
        lines = []
        for _, col, typ, *_ in info:
            try:
                vals = [r[0] for r in con.execute(
                    f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT {n_examples}')]
            except sqlite3.Error:
                vals = []
            ex = ", ".join(repr(v if not isinstance(v, str) else v[:40]) for v in vals)
            lines.append(f"    {_quote(col)} {typ or 'TEXT'}, -- example: [{ex}]")
        if pks:
            lines.append(f"    PRIMARY KEY ({', '.join(_quote(p) for p in pks)}),")
        for fk in fks:
            lines.append(f"    CONSTRAINT fk_{table}_{fk[3]} FOREIGN KEY ({_quote(fk[3])}) "
                         f"REFERENCES {_quote(fk[2])} ({_quote(fk[4] or fk[3])}),")
        if lines:
            lines[-1] = lines[-1].rstrip(",")
        parts.append(f"CREATE TABLE {_quote(table)} (\n" + "\n".join(lines) + "\n);")
    con.close()
    return "\n".join(parts)


OMNISQL = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{schema}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.
"""

AGENT = """You are an assistant that answers questions about a SQLite database by writing SQL.

Database schema:
{schema}

Question: {question}

Write one SQLite query that answers the question. Reply with the query in a ```sql code block."""


def question_text(item: dict) -> str:
    ev = (item.get("evidence") or "").strip()
    return f"{ev}\n{item['question']}" if ev else item["question"]


def build(style: str, item: dict) -> str:
    tpl = OMNISQL if style == "omnisql" else AGENT
    return tpl.format(schema=schema_ddl(item["db_id"]), question=question_text(item))


def extract_sql(text: str) -> str | None:
    """取最后一个 ```sql 代码块；没有代码块时取以 SELECT/WITH 开头的最后一段。"""
    import re
    blocks = re.findall(r"```(?:sql|sqlite)?\s*\n(.*?)```", text or "", flags=re.S | re.I)
    if blocks:
        sql = blocks[-1].strip()
    else:
        m = re.findall(r"((?:SELECT|WITH)\b.*?)(?:;|$)", text or "", flags=re.S | re.I)
        if not m:
            return None
        sql = m[-1].strip()
    lines = [l for l in sql.splitlines() if not l.strip().startswith("--")]
    sql = "\n".join(lines).strip().rstrip(";").strip()
    return sql or None
