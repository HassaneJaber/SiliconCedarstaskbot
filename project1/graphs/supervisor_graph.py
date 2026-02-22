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
    "- conversation: normal chat, explanations\n"
    "- rag: questions about internal docs/kb (docs/*.txt)\n"
    "- viz: user asks for chart/chart.js/visualization/config AND provides data directly\n"
    "- sql: user asks about databases/postgres/tables/SQL OR asks for charts based on DB tables\n"
    "- research: user asks to research, compare, list sources\n"
)

def _wants_chart(text: str) -> bool:
    t = (text or "").lower()
    return any(x in t for x in [
        "chart.js", "chartjs", "bar chart", "pie chart", "line chart",
        "visualize", "plot", "graph", "chart"
    ])

def _looks_like_table_chart(text: str) -> bool:
    t = (text or "").lower()
    return _wants_chart(t) and re.search(r"\bfrom\s+[a-zA-Z_]\w*\b", t) is not None

def _fallback_route(text: str) -> str:
    t = (text or "").lower()

    if _looks_like_table_chart(t):
        return "sql"

    if any(x in t for x in ["postgres", "postgre", "database", "sql", "select", "table", "schema", "query"]):
        return "sql"

    if any(x in t for x in ["research", "sources", "papers", "compare", "survey", "literature", "find articles", "look up"]):
        return "research"

    if _wants_chart(t):
        return "viz"

    if any(x in t for x in ["docs", "kb", "based on docs", "according to the document", "what docs", "list docs"]):
        return "rag"

    return "conversation"


def supervisor_router_node(state: SupervisorState) -> dict:
    llm = get_llm(temperature=0.0)
    last = state["messages"][-1].content

    out = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"USER_MESSAGE:\n{last}")
    ])

    obj = extract_json_object(out.content) or {}
    route = (obj.get("route") or "").strip().lower()
    if route not in {"conversation", "rag", "viz", "research", "sql"}:
        route = _fallback_route(last)

    # Hard override: table-based chart must go SQL first
    if route == "viz" and _looks_like_table_chart(last):
        route = "sql"

    return {"route": route, "error": ""}


def route_fn(state: SupervisorState) -> str:
    return state.get("route", "conversation")


def research_node(state: SupervisorState) -> dict:
    return research_app.invoke(state)


def after_sql_fn(state: SupervisorState) -> str:
    return "viz" if state.get("post_route") == "viz" else "end"


def viz_prep_node(state: SupervisorState) -> dict:
    kind = (state.get("chart_kind") or "bar").strip().lower()
    # This message triggers viz_node but uses the already-fetched sql_preview.
    return {
        "messages": [HumanMessage(content=f"Make a {kind} chart using column 'label' as labels and 'value' as values from the last SQL result.")],
        "route": "viz",
        "error": "",
    }


def build_supervisor_graph():
    g = StateGraph(SupervisorState)

    g.add_node("router", supervisor_router_node)
    g.add_node("conversation", conversation_node)
    g.add_node("rag", rag_node)
    g.add_node("viz", viz_node)
    g.add_node("research", research_node)
    g.add_node("sql", sql_node)
    g.add_node("viz_prep", viz_prep_node)

    g.add_edge(START, "router")

    g.add_conditional_edges(
        "router",
        route_fn,
        {"conversation": "conversation", "rag": "rag", "viz": "viz", "research": "research", "sql": "sql"},
    )

    # SQL -> (optional) VIZ
    g.add_conditional_edges(
        "sql",
        after_sql_fn,
        {"viz": "viz_prep", "end": END},
    )
    g.add_edge("viz_prep", "viz")

    g.add_edge("conversation", END)
    g.add_edge("rag", END)
    g.add_edge("viz", END)
    g.add_edge("research", END)

    if MemorySaver is not None:
        return g.compile(checkpointer=MemorySaver())
    return g.compile()


app = build_supervisor_graph()
graph = app


def run_supervisor(user_text: str, thread_id: str) -> dict:
    res = app.invoke(
        {"messages": [HumanMessage(content=user_text)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    msgs = res.get("messages", [])
    response = msgs[-1].content if msgs else ""
    route = (res.get("route") or "").strip()

    out = {"route": route, "response": response}

    # Helpful extras for Chainlit UI rendering
    for k in ["sql_query", "sql_preview", "viz_config", "chart_kind", "last_agent"]:
        if k in res:
            out[k] = res[k]

    return out