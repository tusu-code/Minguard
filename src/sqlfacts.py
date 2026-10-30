"""确定性层：SQL 解析、输出列血缘、个人数据标注、只读副本上的试执行。

这些都是"硬事实"，由代码计算后交给判官；判官不负责解析和数数。
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, build_scope

import dbmeta

DIALECT = "sqlite"
WRITE_TYPES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Merge,
               exp.TruncateTable, exp.Command, exp.Attach if hasattr(exp, "Attach") else exp.Command)
DANGEROUS_FUNCS = {"load_extension", "readfile", "writefile", "fts3_tokenizer", "edit"}
LINKING = {"identifier", "name", "contact"}   # 能直接把一行连到具体个人的类别
ROW_CAP = 1000
EXEC_TIMEOUT_S = 5.0


@dataclass
class OutCol:
    expr: str
    sources: list[tuple[str, str, bool]]       # [(表, 列, 是否在聚合内)]，原名

    def personal(self, db: str, aggregated: bool | None = None) -> list[tuple[str, str, str]]:
        out = []
        for t, c, agg in self.sources:
            cat = dbmeta.personal_category(db, t, c)
            if cat and (aggregated is None or agg == aggregated):
                out.append((t, c, cat))
        return out


@dataclass
class Facts:
    statement: str                              # select / write / unparsable
    tables: list[str] = field(default_factory=list)
    out_cols: list[OutCol] = field(default_factory=list)
    aggregate_query: bool = False               # 外层有聚合或 GROUP BY
    group_keys: list[tuple[str, str]] = field(default_factory=list)
    distinct: bool = False
    filter_cols: list[tuple[str, str]] = field(default_factory=list)
    dangerous_funcs: list[str] = field(default_factory=list)
    result_rows: int | None = None              # 试执行行数，封顶 ROW_CAP+1
    exec_error: str | None = None
    exec_ms: float | None = None
    parse_error: str | None = None
    contributors: int | None = None             # 聚合查询中参与聚合的不同个人数（最小分组），None=不适用
    conditions: list[str] = field(default_factory=list)   # 顶层 WHERE 合取项 + HAVING + ORDER/LIMIT（v2 事实）

    # ---- 派生事实 ----
    def personal_out(self, db: str) -> list[tuple[str, str, str]]:
        """未聚合地出现在输出里的个人数据列（去重）。"""
        seen, out = set(), []
        for c in self.out_cols:
            for item in c.personal(db, aggregated=False):
                if item[:2] not in seen:
                    seen.add(item[:2])
                    out.append(item)
        return out

    def individual_level(self, db: str) -> bool | None:
        """输出是否含个体级个人数据：True 是 / False 否 / None 边界情形（类别列表等，不进基准）。"""
        pout = self.personal_out(db)
        if not pout:
            return False
        has_link = any(cat in LINKING for _, _, cat in pout)
        if not self.aggregate_query:
            if self.distinct and not has_link:
                return None
            return True
        keys = {(t.lower(), c.lower()) for t, c in self.group_keys}
        if any(dbmeta.personal_category(db, t, c) in LINKING for t, c in self.group_keys):
            return True
        if keys:
            nonkey = [p for p in pout if (p[0].lower(), p[1].lower()) not in keys]
            return True if nonkey else None
        return True     # 无 GROUP BY 的聚合查询里仍有未聚合的个人列（如 MAX(Birthday)）

    def brief(self, db: str) -> dict:
        """交给判官的紧凑事实。只陈述事实，不下策略结论。"""
        cols = []
        for c in self.out_cols:
            srcs = sorted({f"{t}.{k}" for t, k, _ in c.sources})
            pers = c.personal(db)
            item = {"output": c.expr[:90], "derived_from": srcs or ["constant"]}
            if pers:
                item["personal_data"] = sorted({cat for _, _, cat in pers})
                item["aggregated"] = all(a for t, k, a in c.sources if dbmeta.personal_category(db, t, k))
            cols.append(item)
        rows = self.result_rows
        brief = {
            "statement_type": "read-only SELECT" if self.statement == "select" else self.statement,
            "tables_read": self.tables,
            "output_columns": cols,
            "query_aggregates_rows": self.aggregate_query,
            "result_row_count": (f"more than {ROW_CAP}" if rows is not None and rows > ROW_CAP else rows),
        }
        if self.contributors is not None:
            brief["people_aggregated_over"] = self.contributors
        if FACTS_VERSION >= 2:
            brief["row_conditions"] = self.conditions or ["none: every row of the joined tables is returned"]
        if self.exec_error:
            brief["dry_run_error"] = self.exec_error[:120]
        return brief


# ---------------------------------------------------------------- 解析与血缘

def _schema_for(db: str) -> dict:
    return {t: dict(c) for t, c in dbmeta.load_schema(db).items()}


def _agg_inside(node: exp.Expression, stop: exp.Expression) -> bool:
    """node 到 stop 之间是否有（非窗口的）聚合函数。"""
    cur = node.parent
    while cur is not None and cur is not stop.parent:
        # MIN/MAX 返回的是某一个人的值，不算把个体数据聚合掉
        if (isinstance(cur, exp.AggFunc) and not isinstance(cur, (exp.Min, exp.Max))
                and not isinstance(cur.parent, exp.Window)):
            return True
        if cur is stop:
            break
        cur = cur.parent
    return False


def _own_columns(scope: Scope, node: exp.Expression) -> list[exp.Column]:
    """node 中属于本 scope 的列（不下钻子查询）。"""
    own = set(id(c) for c in scope.columns)
    return [c for c in node.find_all(exp.Column) if id(c) in own]


def _sub_scopes_in(scope: Scope, node: exp.Expression) -> list[Scope]:
    out = []
    for s in scope.subquery_scopes:
        e = s.expression
        cur = e.parent
        while cur is not None and cur is not node and cur is not scope.expression:
            cur = cur.parent
        if cur is node:
            out.append(s)
    return out


def _branches(scope: Scope) -> list[Scope]:
    """把 UNION/INTERSECT/EXCEPT 的嵌套二叉 scope 展开成叶子 SELECT 分支。"""
    if scope.is_set_operation and scope.set_operation_scopes:
        out: list[Scope] = []
        for s in scope.set_operation_scopes:
            out.extend(_branches(s))
        return out
    return [scope]


def _resolve_col(scope: Scope, col: exp.Column, depth: int = 0) -> list[tuple[str, str, bool]]:
    if depth > 12:
        return []
    source = scope.sources.get(col.table)
    if isinstance(source, exp.Table):
        return [(source.name, col.name, False)]
    if isinstance(source, Scope):
        branches = _branches(source)
        out: list[tuple[str, str, bool]] = []
        for br in branches:
            sel = br.expression
            if not isinstance(sel, exp.Select):
                continue
            for p in sel.selects:
                if p.alias_or_name == col.name:
                    out.extend(_analyze_proj(br, p, depth + 1))
                    break
        return out
    return []


def _analyze_proj(scope: Scope, proj: exp.Expression, depth: int = 0) -> list[tuple[str, str, bool]]:
    out = []
    for c in _own_columns(scope, proj):
        outer_agg = _agg_inside(c, proj)
        for t, k, inner_agg in _resolve_col(scope, c, depth):
            out.append((t, k, outer_agg or inner_agg))
    # 投影里的标量子查询
    for s in _sub_scopes_in(scope, proj):
        sel = s.expression
        if isinstance(sel, exp.Select) and sel.selects:
            outer_agg = _agg_inside(sel, proj)
            for t, k, a in _analyze_proj(s, sel.selects[0], depth + 1):
                out.append((t, k, outer_agg or a))
    return out


def _is_aggregate_select(sel: exp.Select) -> bool:
    if sel.args.get("group"):
        return True
    for p in sel.selects:
        for a in p.find_all(exp.AggFunc):
            if not isinstance(a.parent, exp.Window):
                # 只算属于本层的聚合
                inner = a.find_ancestor(exp.Select)
                if inner is sel:
                    return True
    return False


def analyze(db: str, sql: str) -> Facts:
    try:
        stmts = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
    except Exception as e:  # noqa: BLE001
        return Facts(statement="unparsable", parse_error=str(e)[:200])
    if len(stmts) != 1:
        kinds = {type(s).__name__ for s in stmts}
        return Facts(statement="multiple statements" if stmts else "unparsable",
                     parse_error=f"{len(stmts)} statements: {sorted(kinds)}")
    tree = stmts[0]
    if isinstance(tree, WRITE_TYPES) or any(isinstance(n, WRITE_TYPES) for n in tree.walk()):
        tables = sorted({t.name for t in tree.find_all(exp.Table)})
        return Facts(statement=f"write ({type(tree).__name__.upper()})", tables=tables)
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)):
        return Facts(statement=f"other ({type(tree).__name__})")

    funcs = sorted({(f.name or "").lower() for f in tree.find_all(exp.Anonymous)} & DANGEROUS_FUNCS)
    try:
        q = qualify(tree.copy(), schema=_schema_for(db), dialect=DIALECT,
                    validate_qualify_columns=False, quote_identifiers=False, identify=False)
        root = build_scope(q)
    except Exception as e:  # noqa: BLE001
        return Facts(statement="select", parse_error=f"qualify: {str(e)[:160]}", dangerous_funcs=funcs)
    if root is None:
        return Facts(statement="select", parse_error="no scope", dangerous_funcs=funcs)

    facts = Facts(statement="select", dangerous_funcs=funcs)
    facts.tables = sorted({t.name for t in q.find_all(exp.Table)})
    branches = _branches(root)
    merged: list[OutCol] = []
    agg_any = False
    for br in branches:
        sel = br.expression
        if not isinstance(sel, exp.Select):
            continue
        agg_any |= _is_aggregate_select(sel)
        facts.distinct |= bool(sel.args.get("distinct"))
        group = sel.args.get("group")
        if group:
            for g in group.expressions:
                for c in _own_columns(br, g):
                    facts.group_keys.extend((t, k) for t, k, _ in _resolve_col(br, c))
        for i, p in enumerate(sel.selects):
            srcs = _analyze_proj(br, p)
            if i < len(merged):
                merged[i].sources.extend(srcs)
            else:
                merged.append(OutCol(expr=p.sql(dialect=DIALECT), sources=srcs))
    facts.out_cols = merged
    facts.aggregate_query = agg_any

    # 过滤列：所有 scope 里出现在 WHERE / HAVING / JOIN ON 的列
    fcols = set()
    for sc in root.traverse():
        for c in sc.columns:
            if c.find_ancestor(exp.Where, exp.Having, exp.Join) is not None:
                for t, k, _ in _resolve_col(sc, c):
                    fcols.add((t, k))
    facts.filter_cols = sorted(fcols)
    return facts


# ---------------------------------------------------------------- 试执行

def dry_run(db: str, sql: str, facts: Facts | None = None) -> tuple[list[tuple] | None, Facts | None]:
    """只读连接上执行，最多取 ROW_CAP+1 行；超时中断。返回 (rows, facts)。"""
    con = dbmeta.connect_ro(db)
    deadline = time.monotonic() + EXEC_TIMEOUT_S
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    t0 = time.monotonic()
    rows = None
    err = None
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(ROW_CAP + 1)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    finally:
        con.set_progress_handler(None, 0)
    if facts is not None:
        facts.exec_ms = (time.monotonic() - t0) * 1000
        facts.exec_error = err
        facts.result_rows = None if rows is None else len(rows)
    return rows, facts


# 每张"人"表的个人标识列：用来数聚合背后有几个人
PERSON_KEY = {"Patient": "ID", "Examination": "ID", "Laboratory": "ID", "member": "member_id",
              "client": "client_id", "disp": "client_id", "account": "account_id", "loan": "account_id",
              "trans": "account_id", "users": "Id", "income": "link_to_member", "expense": "link_to_member",
              "attendance": "link_to_member", "drivers": "driverId", "results": "driverId",
              "qualifying": "driverId", "lapTimes": "driverId", "pitStops": "driverId", "driverStandings": "driverId",
              "Player": "player_api_id", "Player_Attributes": "player_api_id"}
SMALL_GROUP = 3
FACTS_VERSION = int(__import__("os").environ.get("FACTS_VERSION", "1"))


def top_conditions(sql: str) -> list[str]:
    """顶层查询决定"返回哪些行"的条件：WHERE 的每个 AND 项、HAVING、ORDER BY + LIMIT。"""
    try:
        tree = sqlglot.parse_one(sql, read=DIALECT)
    except Exception:  # noqa: BLE001
        return []
    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if sel is None:
        return []
    out = []
    w = sel.args.get("where")
    if w is not None:
        parts, stack = [], [w.this]
        while stack:
            n = stack.pop()
            if isinstance(n, exp.And):
                stack.extend([n.right, n.left])
            else:
                parts.append(n)
        out += [f"WHERE {p.sql(dialect=DIALECT)[:120]}" for p in parts]
    if sel.args.get("having") is not None:
        out.append(f"HAVING {sel.args['having'].this.sql(dialect=DIALECT)[:120]}")
    lim = sel.args.get("limit")
    if lim is not None:
        order = sel.args.get("order")
        o = f" ordered by {order.sql(dialect=DIALECT)[9:80]}" if order is not None else ""
        out.append(f"LIMIT {lim.expression.sql(dialect=DIALECT) if lim.expression else lim.sql()}{o}")
    for j in sel.args.get("joins") or []:
        on = j.args.get("on")
        if on is not None and any(isinstance(x, exp.Literal) for x in on.find_all(exp.Literal)):
            out.append(f"JOIN ON {on.sql(dialect=DIALECT)[:120]}")
    return out[:8]


def contributors(db: str, sql: str) -> int | None:
    """顶层聚合查询（含个人表）背后最小分组的不同个人数。

    改写：保留原 FROM/JOIN/WHERE/GROUP BY，把投影换成 COUNT(DISTINCT 个人标识)，取各组最小值。
    只处理顶层 SELECT；拿不到个人表或改写失败返回 None。
    """
    try:
        tree = sqlglot.parse_one(sql, read=DIALECT)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(tree, exp.Select):
        return None
    frm = tree.args.get("from") or tree.args.get("from_")
    nodes = ([frm.this] if frm is not None else []) + [j.this for j in tree.args.get("joins") or []]
    key = None
    for n in nodes:
        if isinstance(n, exp.Table):
            t = next((k for k in PERSON_KEY if k.lower() == n.name.lower()), None)
            if t:
                key = f'{n.alias_or_name}."{PERSON_KEY[t]}"'
                break
    if key is None:
        return None
    new = tree.copy()
    new.set("expressions", [sqlglot.parse_one(f"COUNT(DISTINCT {key})", read=DIALECT).as_("n")])
    for k in ("order", "limit", "offset", "having", "distinct"):
        new.set(k, None)
    q = f"SELECT MIN(n) FROM ({new.sql(dialect=DIALECT)})"
    rows, _ = dry_run(db, q)
    if not rows or rows[0][0] is None:
        return None
    return int(rows[0][0])


def facts_for(db: str, sql: str, execute: bool = True) -> Facts:
    f = analyze(db, sql)
    if f.statement == "select" and not f.parse_error:
        f.conditions = top_conditions(sql)
    if execute and f.statement == "select" and not f.parse_error:
        dry_run(db, sql, f)
        if f.aggregate_query and not f.exec_error:
            f.contributors = contributors(db, sql)
    return f
