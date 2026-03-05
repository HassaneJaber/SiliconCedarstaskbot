import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from project1.core.llm import get_llm
from project1.core.state import SupervisorState
from project1.core.text_utils import extract_json_object

try:
    import psycopg2
    import psycopg2.extras
except Exception:
    psycopg2 = None


SQL_GEN_SYSTEM = (
    "You are a SQL Query Agent for PostgreSQL.\n"
    "Return JSON only: {\"sql\": \"...\"}\n"
    "Rules:\n"
    "- ONLY read-only queries (SELECT or WITH ... SELECT)\n"
    "- Prefer LIMIT 50 if not provided\n"
    "- Do NOT use INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE\n"
    "- If schema is unknown, use information_schema to inspect tables/columns.\n"
)

SQL_VIZ_GEN_SYSTEM = (
    "You are a SQL Query Agent for PostgreSQL.\n"
    "The user wants to VISUALIZE results.\n"
    "Return JSON only: {\"sql\": \"...\"}\n"
    "Rules:\n"
    "- ONLY read-only queries (SELECT or WITH ... SELECT)\n"
    "- Prefer LIMIT 50 if not provided\n"
    "- Do NOT use INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/TRUNCATE\n"
    "- Output MUST be chart-friendly:\n"
    "  * exactly 2 columns: label (text) + value (numeric)\n"
    "  * Use aliases: label, value\n"
    "  * Example: SELECT name AS label, age AS value FROM demo_users LIMIT 50\n"
)

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke)\b",
    re.IGNORECASE,
)

CHART_RE = re.compile(r"\b(chart|chart\.js|plot|graph|visuali[sz]e)\b", re.IGNORECASE)


def _get_dsn() -> str:
    return (os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL") or "").strip()


def _first_statement(sql: str) -> str:
    parts = [p.strip() for p in (sql or "").split(";") if p.strip()]
    return parts[0] if parts else ""


def _is_safe_sql(sql: str) -> bool:
    s = (sql or "").strip()
    if not s:
        return False
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return False
    if FORBIDDEN.search(low):
        return False
    return True


def _ensure_limit(sql: str, limit: int = 50) -> str:
    low = sql.lower()
    if re.search(r"\blimit\b", low):
        return sql
    return sql.rstrip() + f"\nLIMIT {limit}"


def _to_md_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_(no rows)_"
    cols = list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join([header, sep] + body)


def _run_query(dsn: str, sql: str, max_rows: int = 50) -> List[Dict[str, Any]]:
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            try:
                rows = cur.fetchmany(max_rows)
            except Exception:
                rows = []
        return rows
    finally:
        conn.close()


def _chart_kind(text: str) -> str:
    t = (text or "").lower()
    if "doughnut" in t or "donut" in t:
        return "doughnut"
    if "pie" in t:
        return "pie"
    if "line" in t:
        return "line"
    if "bar" in t:
        return "bar"
    return "bar"


def _extract_requested_metric(text: str) -> str:
    t = (text or "").lower()
    if "age" in t or "ages" in t:
        return "age"
    if "count" in t or "how many" in t or "number of" in t:
        return "count"
    if "avg" in t or "average" in t or "mean" in t:
        return "avg"
    if "sum" in t or "total" in t:
        return "sum"
    return ""


def _safe_ident(name: str) -> Optional[str]:
    if not name:
        return None
    name = name.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return name
    return None


def _guess_table_from_text(dsn: str, text: str) -> str:
    m = re.search(r"\bfrom\s+([a-zA-Z_]\w*)\b", text, re.IGNORECASE)
    if m:
        return m.group(1)

    rows = _run_query(
        dsn,
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public' LIMIT 200",
        max_rows=200,
    )
    names = [r["table_name"] for r in rows if "table_name" in r]
    low = (text or "").lower()
    for name in names:
        if name.lower() in low:
            return name
    return ""


def _get_columns(dsn: str, table: str) -> List[Dict[str, Any]]:
    sql = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema='public' AND table_name=%s
    ORDER BY ordinal_position
    """
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (table,))
            return cur.fetchall() or []
    finally:
        conn.close()


def _pick_label_value(columns: List[Dict[str, Any]], metric_hint: str) -> Tuple[str, str]:
    names = [c["column_name"] for c in columns]
    types = {c["column_name"]: (c.get("data_type") or "").lower() for c in columns}

    def is_numeric(col: str) -> bool:
        dt = types.get(col, "")
        return any(x in dt for x in ["int", "numeric", "double", "real", "decimal", "bigint", "smallint"])

    def is_text(col: str) -> bool:
        dt = types.get(col, "")
        return any(x in dt for x in ["text", "character", "varchar", "char"])

    # Label preference
    label_col = ""
    for preferred in ["name", "title", "label"]:
        if preferred in names and is_text(preferred):
            label_col = preferred
            break
    if not label_col:
        text_cols = [c for c in names if is_text(c)]
        label_col = text_cols[0] if text_cols else ""

    # Value preference
    value_col = ""
    if metric_hint and metric_hint in names and is_numeric(metric_hint):
        value_col = metric_hint
    else:
        numeric_cols = [c for c in names if is_numeric(c)]
        non_id = [c for c in numeric_cols if c.lower() != "id" and not c.lower().endswith("_id")]
        value_col = non_id[0] if non_id else (numeric_cols[0] if numeric_cols else "")

    if not label_col and "id" in names:
        label_col = "id"

    return label_col, value_col


def _demo_users_sql_from_text(user_text: str) -> Optional[str]:
    """
    Deterministic shortcuts so the user can ask normal questions
    without knowing SQL.
    """
    t = (user_text or "").lower()

    if CHART_RE.search(t) and ("age" in t or "ages" in t) and ("user" in t or "demo_users" in t):
        return "SELECT name AS label, age AS value FROM public.demo_users LIMIT 50"

    if re.search(r"\b(show|list|display|get)\b.*\b(all\s+)?users\b", t) or "who are the users" in t:
        return "SELECT * FROM public.demo_users LIMIT 50"

    if ("name" in t or "names" in t) and ("age" in t or "ages" in t) and ("user" in t or "demo_users" in t):
        return "SELECT name, age FROM public.demo_users LIMIT 50"

    if "average age" in t or "avg age" in t or "mean age" in t:
        return "SELECT ROUND(AVG(age)::numeric, 2) AS average_age FROM public.demo_users"

    if "how many users" in t or "count users" in t or "number of users" in t:
        return "SELECT COUNT(*) AS user_count FROM public.demo_users"

    if "oldest user" in t or "oldest users" in t:
        return "SELECT * FROM public.demo_users ORDER BY age DESC LIMIT 1"

    if "youngest user" in t or "youngest users" in t:
        return "SELECT * FROM public.demo_users ORDER BY age ASC LIMIT 1"

    if "demo_users" in t and any(x in t for x in ["show", "list", "display", "get"]):
        return "SELECT * FROM public.demo_users LIMIT 50"

    return None


def _answer_from_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "I couldn't find any matching rows."

    if len(rows) == 1:
        row = rows[0]

        if "user_count" in row:
            return f"There are {row['user_count']} users."

        if "average_age" in row:
            return f"The average age is {row['average_age']}."

        if len(row) == 1:
            only_val = next(iter(row.values()))
            return f"The answer is {only_val}."

    return "Here are the results:\n\n" + _to_md_table(rows)


def sql_node(state: SupervisorState) -> dict:
    user_text = state["messages"][-1].content.strip()

    if psycopg2 is None:
        return {
            "messages": [AIMessage(content="psycopg2 is not installed. Run: python -m pip install psycopg2-binary")],
            "last_agent": "sql",
            "error": "",
            "post_route": "",
        }

    dsn = _get_dsn()
    if not dsn:
        return {
            "messages": [AIMessage(content="PostgreSQL is not configured. Add POSTGRES_DSN in .env (postgresql://user:pass@host:5432/dbname).")],
            "last_agent": "sql",
            "error": "",
            "post_route": "",
        }

    wants_chart = bool(CHART_RE.search(user_text))
    chart_kind = _chart_kind(user_text)
    metric_hint = _extract_requested_metric(user_text)

    # 1) If the user typed direct SQL, allow it (safe read-only only)
    direct = _first_statement(user_text)
    if _is_safe_sql(direct):
        sql = _ensure_limit(direct, 50)
        auto_viz = False

    else:
        # 2) Deterministic natural-language shortcuts for demo_users
        demo_sql = _demo_users_sql_from_text(user_text)
        if demo_sql:
            sql = _ensure_limit(demo_sql, 50)
            auto_viz = wants_chart

        # 3) Chart requests: try to build chart-friendly query
        elif wants_chart:
            table = _safe_ident(_guess_table_from_text(dsn, user_text))
            sql = ""

            if table:
                cols = _get_columns(dsn, table)
                label_col, value_col = _pick_label_value(cols, metric_hint)

                if label_col and value_col:
                    if label_col == "id":
                        sql = f"SELECT CAST({label_col} AS TEXT) AS label, {value_col} AS value FROM {table} LIMIT 50"
                    else:
                        sql = f"SELECT {label_col} AS label, {value_col} AS value FROM {table} LIMIT 50"

            if not _is_safe_sql(sql):
                llm = get_llm(temperature=0.0)
                out = llm.invoke([
                    SystemMessage(content=SQL_VIZ_GEN_SYSTEM),
                    HumanMessage(content=user_text),
                ])
                obj = extract_json_object(out.content) or {}
                sql = _first_statement(str(obj.get("sql", "")).strip())

            if not _is_safe_sql(sql):
                sql = """
SELECT table_schema, table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema NOT IN ('pg_catalog','information_schema')
ORDER BY table_schema, table_name, ordinal_position
LIMIT 50
""".strip()

            sql = _ensure_limit(sql, 50)
            auto_viz = True

        # 4) General natural language -> let the LLM make SQL
        else:
            llm = get_llm(temperature=0.0)
            out = llm.invoke([
                SystemMessage(content=SQL_GEN_SYSTEM),
                HumanMessage(content=user_text),
            ])
            obj = extract_json_object(out.content) or {}
            sql = _first_statement(str(obj.get("sql", "")).strip())

            if not _is_safe_sql(sql):
                sql = """
SELECT table_schema, table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema NOT IN ('pg_catalog','information_schema')
ORDER BY table_schema, table_name, ordinal_position
LIMIT 50
""".strip()

            sql = _ensure_limit(sql, 50)
            auto_viz = False

    try:
        preview_rows = _run_query(dsn, sql, max_rows=50)

        if auto_viz:
            # SQL is only preparing data for viz; final answer will come from viz_node
            answer = "I prepared the data for the chart."
        else:
            answer = _answer_from_rows(preview_rows)

        return {
            "messages": [AIMessage(content=answer)],
            "last_agent": "sql",
            "sql_query": sql,          # keep internal only
            "sql_preview": preview_rows,
            "chart_kind": chart_kind,
            "post_route": ("viz" if auto_viz else ""),
            "error": "",
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"SQL error: {type(e).__name__}: {e}")],
            "last_agent": "sql",
            "post_route": "",
            "error": "",
        }