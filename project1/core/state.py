from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SupervisorState(TypedDict, total=False):
    # Chat history
    messages: Annotated[List[BaseMessage], add_messages]

    # Routing / bookkeeping
    route: str
    last_agent: str
    error: str

    # Generic same-run handoff
    handoff_to: str   # "", "conversation", "rag", "sql", "viz", "research"

    # SQL outputs
    sql_query: str
    sql_preview: List[Dict[str, Any]]

    # SQL -> VIZ auto-handoff
    post_route: str      # "viz" or ""
    chart_kind: str      # "bar" | "line" | "pie" | "doughnut"

    # VIZ output
    viz_config: Dict[str, Any]

    # Research output (optional)
    research_notes: Dict[str, Any]