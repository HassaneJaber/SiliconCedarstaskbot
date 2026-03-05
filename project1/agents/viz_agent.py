import os
import re
import json
from typing import Any, Dict, List, Tuple, Optional

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_core.messages import AIMessage
from project1.core.state import SupervisorState

try:
    import psycopg2
    import psycopg2.extras
except Exception:
    psycopg2 = None


CHART_KEYWORDS = re.compile(r"\b(chart|plot|graph|visualize|visualisation|visualization)\b", re.I)


def _get_dsn() -> str:
    return (os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL") or "").strip()


def _safe_table_name(name: str) -> Optional[str]:
    if not name:
        return None
    name = name.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return name
    return None


def _parse_kv_pairs(text: str) -> List[Tuple[str, float]]:
    pairs = []
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)", text):
        pairs.append((m.group(1), float(m.group(2))))
    return pairs


def _parse_labels_values(text: str) -> Optional[Tuple[List[str], List[float]]]:
    """
    Supports prompts like:
    - bar chart with labels A, B, C and values 10, 20, 30
    - labels: A,B,C values: 10,20,30
    """
    m = re.search(
        r"\blabels?\s*(?:=|:)?\s*([A-Za-z0-9_,\s]+?)\s*(?:and\s+)?values?\s*(?:=|:)?\s*([-\d\.,\s]+)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None

    raw_labels = m.group(1).strip()
    raw_values = m.group(2).strip()

    labels = [x.strip() for x in re.split(r"\s*,\s*", raw_labels) if x.strip()]
    if len(labels) <= 1:
        labels = [x.strip() for x in re.split(r"\s+", raw_labels) if x.strip()]

    values = []
    for x in re.split(r"[,\s]+", raw_values):
        if not x:
            continue
        try:
            values.append(float(x))
        except Exception:
            return None

    if len(labels) >= 2 and len(labels) == len(values):
        return labels, values

    return None


def _parse_points_list(text: str) -> Optional[List[float]]:
    m = re.search(r"\b(points|values|data)\s*:\s*([-\d\.\s,]+)", text, re.I)
    if not m:
        return None
    raw = m.group(2).strip()
    nums = re.split(r"[,\s]+", raw)
    out = []
    for x in nums:
        if not x:
            continue
        try:
            out.append(float(x))
        except Exception:
            return None
    return out if out else None


def _detect_chart_type(text: str, default_type: str) -> str:
    t = (text or "").lower()
    if "doughnut" in t or "donut" in t:
        return "doughnut"
    if "pie" in t:
        return "pie"
    if "bar" in t:
        return "bar"
    if "line" in t:
        return "line"
    return default_type


def _chart_config(chart_type: str, labels: List[str], values: List[float], dataset_label: str = "Value") -> Dict[str, Any]:
    return {
        "type": chart_type,
        "data": {
            "labels": labels,
            "datasets": [{"label": dataset_label, "data": values}],
        },
        "options": {
            "responsive": True,
            "plugins": {"legend": {"display": True}, "title": {"display": False, "text": ""}},
        },
    }


def _infer_from_rows(rows: List[Dict[str, Any]]) -> Optional[Tuple[List[str], List[float], str]]:
    if not rows:
        return None
    cols = list(rows[0].keys())
    if not cols:
        return None

    label_col = None
    for c in cols:
        if isinstance(rows[0].get(c), str):
            label_col = c
            break
    if label_col is None:
        label_col = cols[0]

    value_col = None
    for c in cols:
        if c == label_col:
            continue
        if isinstance(rows[0].get(c), (int, float)):
            value_col = c
            break
    if value_col is None:
        if len(cols) >= 2:
            value_col = cols[1]
        else:
            return None

    labels = [str(r.get(label_col, "")) for r in rows]
    values = []
    for r in rows:
        try:
            values.append(float(r.get(value_col, 0)))
        except Exception:
            values.append(0.0)

    return labels, values, "Value"


def _query_table_preview(table: str, limit: int = 50) -> Optional[List[Dict[str, Any]]]:
    if psycopg2 is None:
        return None
    dsn = _get_dsn()
    if not dsn:
        return None
    sql = f"SELECT * FROM {table} LIMIT {limit}"
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return cur.fetchmany(50)
    finally:
        conn.close()


def viz_node(state: SupervisorState) -> dict:
    user_text = state["messages"][-1].content.strip()

    if not CHART_KEYWORDS.search(user_text):
        return {
            "messages": [AIMessage(content="Tell me what chart you want (bar/line/pie) and the data.")],
            "last_agent": "viz",
            "error": "",
        }

    chart_type = _detect_chart_type(user_text, default_type="bar")

    # 1) Direct user-provided key=value pairs should ALWAYS win
    pairs = _parse_kv_pairs(user_text)
    if len(pairs) >= 2:
        labels = [k for k, _ in pairs]
        values = [v for _, v in pairs]
        cfg = _chart_config(chart_type if chart_type else "pie", labels, values, dataset_label="Value")
        return {
            "messages": [AIMessage(content=f"Here is your {chart_type} chart.")],
            "last_agent": "viz",
            "viz_config": cfg,
            "error": "",
        }

    # 2) Explicit labels + values format
    lv = _parse_labels_values(user_text)
    if lv:
        labels, values = lv
        cfg = _chart_config(chart_type, labels, values, dataset_label="Value")
        return {
            "messages": [AIMessage(content=f"Here is your {chart_type} chart.")],
            "last_agent": "viz",
            "viz_config": cfg,
            "error": "",
        }

    # 3) Explicit points list
    pts = _parse_points_list(user_text)
    if pts:
        labels = [str(i + 1) for i in range(len(pts))]
        cfg = _chart_config(chart_type if chart_type else "line", labels, pts, dataset_label="points")
        return {
            "messages": [AIMessage(content=f"Here is your {chart_type} chart.")],
            "last_agent": "viz",
            "viz_config": cfg,
            "error": "",
        }

    # 4) Use sql_preview ONLY in the same SQL -> viz handoff turn
    use_sql_preview = state.get("post_route") == "viz"
    rows = state.get("sql_preview") or []
    if use_sql_preview and rows:
        inferred = _infer_from_rows(rows)
        if inferred:
            labels, values, ds_label = inferred
            cfg = _chart_config(chart_type, labels, values, dataset_label=ds_label)
            return {
                "messages": [AIMessage(content=f"Here is your {chart_type} chart.")],
                "last_agent": "viz",
                "viz_config": cfg,
                "error": "",
            }

    # 5) If user mentions a table directly, fetch it now
    table = None
    m1 = re.search(r"\bfrom\s+([A-Za-z_][A-Za-z0-9_]*)\b", user_text, re.I)
    m2 = re.search(r"\bin\s+([A-Za-z_][A-Za-z0-9_]*)\b", user_text, re.I)
    cand = (m1.group(1) if m1 else None) or (m2.group(1) if m2 else None)
    cand = _safe_table_name(cand) if cand else None

    if cand:
        fetched = _query_table_preview(cand)
        inferred2 = _infer_from_rows(fetched or [])
        if inferred2:
            labels, values, ds_label = inferred2
            cfg = _chart_config(chart_type, labels, values, dataset_label=ds_label)
            return {
                "messages": [AIMessage(content=f"Here is your {chart_type} chart.")],
                "last_agent": "viz",
                "viz_config": cfg,
                "sql_preview": fetched,
                "error": "",
            }

    return {
        "messages": [AIMessage(content="I couldn't detect chart data. Try asking for a chart and include values.")],
        "last_agent": "viz",
        "error": "",
    }