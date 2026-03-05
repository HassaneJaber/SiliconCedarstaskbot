# project1/graphs/supervisor_graph.py
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception:
    MemorySaver = None

from project1.core.llm import get_llm
from project1.core.state import SupervisorState
from project1.core.text_utils import extract_json_object

from project1.agents.conversation_agent import conversation_node
from project1.agents.rag_agent import rag_node
from project1.agents.viz_agent import viz_node
from project1.agents.sql_agent import sql_node
from project1.agents.research_team import research_app


ROUTER_SYSTEM = (
    "You are the Main Supervisor.\n"
    "Choose the best route for the user's last message.\n"
    "Return JSON only: {\"route\": \"conversation|rag|viz|research|sql\"}\n"
    "Guidance:\n"
    "- conversation: normal chat, explanations, and questions about how this app/system works\n"
    "- rag: questions about internal docs/kb (docs/*.txt)\n"
    "- viz: user asks for chart/chart.js/visualization/config AND provides data directly\n"
    "- sql: user asks about databases/postgres/tables/SQL OR asks for charts based on DB tables\n"
    "- research: user asks to research, compare, list sources\n"
)

SQL_DIRECT_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def _wants_chart(text: str) -> bool:
    t = (text or "").lower()
    return any(
        x in t
        for x in [
            "chart.js",
            "chartjs",
            "bar chart",
            "pie chart",
            "line chart",
            "visualize",
            "visualise",
            "plot",
            "graph",
            "chart",
        ]
    )


def _looks_like_table_chart(text: str) -> bool:
    t = (text or "").lower()
    return _wants_chart(t) and re.search(r"\bfrom\s+[a-zA-Z_]\w*\b", t) is not None


def _has_inline_chart_data(text: str) -> bool:
    t = text or ""
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*-?\d+(?:\.\d+)?", t):
        return True
    if re.search(r"\b(points|values|data)\s*:\s*[-\d\.,\s]+", t, re.IGNORECASE):
        return True
    if re.search(
        r"\blabels?\s*(?:=|:)?\s*[A-Za-z0-9_,\s]+\s*(?:and\s+)?values?\s*(?:=|:)?\s*[-\d\.,\s]+",
        t,
        re.IGNORECASE,
    ):
        return True
    return False


def _is_app_behavior_question(text: str) -> bool:
    t = (text or "").lower()
    return any(
        k in t
        for k in [
            "thread_id",
            "thread id",
            "checkpoint",
            "checkpointer",
            "route",
            "routing",
            "router",
            "supervisor",
            "chainlit",
            "session",
            "why did it choose",
            "based on what does it classify",
            "how does it decide",
            "how does the routing work",
            "how does this app work",
            "in this app",
            "in this system",
            "what is the role of langgraph here",
            "what is the actual role of langgraph here",
            "what does memory mean in this app",
        ]
    )


def _hard_route(text: str) -> str:
    t = (text or "").lower().strip()

    # 0) App behavior/meta questions -> conversation
    if _is_app_behavior_question(t):
        return "conversation"

    # 1) Direct SQL should always go SQL
    if SQL_DIRECT_RE.match(t):
        return "sql"

    # 2) Known internal-doc topics -> RAG (keep this, but app-behavior override already handled above)
    if any(
        x in t
        for x in [
            "langgraph",
            "langsmith",
            "docs/kb",
            "internal docs",
            "according to the docs",
            "based on the docs",
            "what docs",
            "list docs",
            "show docs",
            "which documents",
        ]
    ):
        return "rag"

    # 3) Research keywords
    if any(
        x in t
        for x in [
            "research",
            "sources",
            "papers",
            "compare",
            "survey",
            "literature",
            "find articles",
            "look up",
        ]
    ):
        return "research"

    # 4) Chart from DB table -> SQL first
    if _looks_like_table_chart(t):
        return "sql"

    # 5) Direct chart with inline user data -> viz
    if _wants_chart(t) and _has_inline_chart_data(t):
        return "viz"

    # 6) Database/business data requests
    if any(
        x in t
        for x in [
            "postgres",
            "postgre",
            "database",
            "sql",
            "select",
            "table",
            "schema",
            "query",
            "price",
            "stock",
            "inventory",
            "users",
            "demo_users",
        ]
    ):
        return "sql"

    # 7) Generic chart request
    if _wants_chart(t):
        return "viz"

    return ""


def _fallback_route(text: str) -> str:
    hard = _hard_route(text)
    return hard if hard else "conversation"


def supervisor_router_node(state: SupervisorState) -> dict:
    last = state["messages"][-1].content

    # Deterministic routing first
    hard = _hard_route(last)
    if hard:
        return {"route": hard, "handoff_to": "", "error": ""}

    llm = get_llm(temperature=0.0)
    out = llm.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"USER_MESSAGE:\n{last}"),
        ]
    )

    obj = extract_json_object(out.content) or {}
    route = (obj.get("route") or "").strip().lower()

    if route not in {"conversation", "rag", "viz", "research", "sql"}:
        route = _fallback_route(last)

    # Hard override 1: app-behavior questions should NOT go to RAG
    if _is_app_behavior_question(last):
        route = "conversation"

    # Hard override 2: table-based chart must go SQL first
    if route == "viz" and _looks_like_table_chart(last):
        route = "sql"

    return {"route": route, "handoff_to": "", "error": ""}


def route_fn(state: SupervisorState) -> str:
    return state.get("route", "conversation")


def research_node(state: SupervisorState) -> dict:
    # research_app already contains its own internal multi-step flow
    result = research_app.invoke(state)
    # ensure no accidental loop
    result["handoff_to"] = ""
    result["route"] = result.get("route") or "research"
    return result


def after_sql_fn(state: SupervisorState) -> str:
    # Keep your existing SQL -> VIZ same-run flow
    return "viz_prep" if state.get("post_route") == "viz" else "handoff"


def viz_prep_node(state: SupervisorState) -> dict:
    kind = (state.get("chart_kind") or "bar").strip().lower()
    return {
        "messages": [
            HumanMessage(
                content=(
                    f"Make a {kind} chart using column 'label' as labels and "
                    f"'value' as values from the last SQL result."
                )
            )
        ],
        "route": "viz",
        "handoff_to": "",
        "error": "",
    }


def handoff_node(state: SupervisorState) -> dict:
    # keep state as-is
    return {}


def handoff_route_fn(state: SupervisorState) -> str:
    """
    Decide where to go next *within the same run* based on handoff_to.
    IMPORTANT: If we do a handoff, update `route` to the destination so the UI
    shows the final handed-off route (conversation/rag/viz/research/sql).
    """
    target = (state.get("handoff_to") or "").strip().lower()
    if target in {"conversation", "rag", "viz", "research", "sql"}:
        return target
    return "end"


def _set_route_for_handoff(state: SupervisorState, dest: str) -> dict:
    """
    Ensure route reflects the handoff destination so the UI prints the final route.
    """
    return {"route": dest, "error": ""}


def build_supervisor_graph():
    g = StateGraph(SupervisorState)

    g.add_node("router", supervisor_router_node)
    g.add_node("conversation", conversation_node)
    g.add_node("rag", rag_node)
    g.add_node("viz", viz_node)
    g.add_node("research", research_node)
    g.add_node("sql", sql_node)
    g.add_node("viz_prep", viz_prep_node)
    g.add_node("handoff", handoff_node)

    # helper nodes to update route when handing off
    g.add_node("set_route_conversation", lambda s: _set_route_for_handoff(s, "conversation"))
    g.add_node("set_route_rag", lambda s: _set_route_for_handoff(s, "rag"))
    g.add_node("set_route_viz", lambda s: _set_route_for_handoff(s, "viz"))
    g.add_node("set_route_research", lambda s: _set_route_for_handoff(s, "research"))
    g.add_node("set_route_sql", lambda s: _set_route_for_handoff(s, "sql"))

    g.add_edge(START, "router")

    # Initial supervisor dispatch
    g.add_conditional_edges(
        "router",
        route_fn,
        {
            "conversation": "conversation",
            "rag": "rag",
            "viz": "viz",
            "research": "research",
            "sql": "sql",
        },
    )

    # SQL -> (optional) VIZ, otherwise generic handoff logic
    g.add_conditional_edges(
        "sql",
        after_sql_fn,
        {"viz_prep": "viz_prep", "handoff": "handoff"},
    )
    g.add_edge("viz_prep", "viz")

    # All workers return to the generic handoff node
    g.add_edge("conversation", "handoff")
    g.add_edge("rag", "handoff")
    g.add_edge("viz", "handoff")
    g.add_edge("research", "handoff")

    # Generic dynamic handoff within the same run
    # We first update the route, then jump to the destination agent.
    g.add_conditional_edges(
        "handoff",
        handoff_route_fn,
        {
            "conversation": "set_route_conversation",
            "rag": "set_route_rag",
            "viz": "set_route_viz",
            "research": "set_route_research",
            "sql": "set_route_sql",
            "end": END,
        },
    )

    g.add_edge("set_route_conversation", "conversation")
    g.add_edge("set_route_rag", "rag")
    g.add_edge("set_route_viz", "viz")
    g.add_edge("set_route_research", "research")
    g.add_edge("set_route_sql", "sql")

    if MemorySaver is not None:
        return g.compile(checkpointer=MemorySaver())
    return g.compile()


app = build_supervisor_graph()
graph = app


def run_supervisor(user_text: str, thread_id: str) -> dict:
    # Clear one-turn transient fields on every new user turn
    res = app.invoke(
        {
            "messages": [HumanMessage(content=user_text)],
            "route": "",
            "error": "",
            "handoff_to": "",
            "sql_query": "",
            "sql_preview": [],
            "viz_config": None,
            "chart_kind": "",
            "post_route": "",
            "last_agent": "",
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    msgs = res.get("messages", [])
    response = msgs[-1].content if msgs else ""
    route = (res.get("route") or "").strip() or "conversation"

    out = {"route": route, "response": response}

    # Do NOT expose SQL query text to the UI
    for k in ["sql_preview", "viz_config", "chart_kind", "last_agent"]:
        if k in res:
            out[k] = res[k]

    return out