from dotenv import load_dotenv
load_dotenv()

import os
import re
from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
llm = ChatGroq(model=MODEL, temperature=0)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    calc_result: str


# ----------------------------
# Tool: safe calculator
# ----------------------------
ALLOWED = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]+$")

def safe_calc(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return "Error: empty expression."
    if not ALLOWED.match(expr):
        return "Error: expression contains invalid characters."
    try:
        # Evaluate with no builtins
        val = eval(expr, {"__builtins__": {}}, {})
        return str(val)
    except Exception as e:
        return f"Error: {e}"


def route_node(state: State) -> dict:
    """Decide whether to use calculator or answer directly."""
    user_text = state["messages"][-1].content

    sys = SystemMessage(content=(
        "Decide the route.\n"
        "If the user asks for math or contains a math expression, output ONLY: calc\n"
        "Otherwise output ONLY: direct"
    ))
    resp = llm.invoke([sys, HumanMessage(content=user_text)])
    decision = resp.content.strip().lower()

    route = "calc" if "calc" in decision else "direct"
    return {"route": route}


def calc_node(state: State) -> dict:
    """Extract expression and compute it."""
    user_text = state["messages"][-1].content

    # very simple extraction: take everything after 'calc:' if present, else use whole text
    expr = user_text.split("calc:", 1)[1] if "calc:" in user_text.lower() else user_text

    result = safe_calc(expr)
    return {"calc_result": result}


def answer_with_calc_node(state: State) -> dict:
    """Return final answer after calculator."""
    result = state.get("calc_result", "")
    user_text = state["messages"][-1].content

    sys = SystemMessage(content="You are a helpful assistant. If a calculator result is provided, use it.")
    msg = HumanMessage(content=f"User asked: {user_text}\nCalculator result: {result}\nReply with the final answer.")
    resp = llm.invoke([sys, msg])

    return {"messages": [resp]}


def direct_answer_node(state: State) -> dict:
    sys = SystemMessage(content="You are a helpful assistant.")
    resp = llm.invoke([sys] + state["messages"])
    return {"messages": [resp]}


def route_fn(state: State) -> str:
    return state["route"]


# ----------------------------
# Build graph
# ----------------------------
builder = StateGraph(State)

builder.add_node("route", route_node)
builder.add_node("calc", calc_node)
builder.add_node("answer_with_calc", answer_with_calc_node)
builder.add_node("direct", direct_answer_node)

builder.add_edge(START, "route")
builder.add_conditional_edges("route", route_fn, {
    "calc": "calc",
    "direct": "direct"
})

builder.add_edge("calc", "answer_with_calc")
builder.add_edge("answer_with_calc", END)
builder.add_edge("direct", END)

graph = builder.compile(checkpointer=InMemorySaver())


def main():
    thread_id = "tool-thread-1"
    print("\nExercise 4: LangGraph tool routing (calculator)")
    print("Try: '13*13+1' or 'calc: (8*12)+12' or a normal question.\nType 'exit' to stop.\n")

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"tool_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with open(log_path, "w", encoding="utf-8") as log:
        while True:
            user = input("You: ").strip()
            if user.lower() in ("exit", "quit"):
                break

            result = graph.invoke(
                {"messages": [HumanMessage(content=user)]},
                {"configurable": {"thread_id": thread_id}},
            )

            answer = result["messages"][-1].content
            route = result.get("route", "?")
            calc_result = result.get("calc_result", "")

            print(f"\n[route={route} | calc_result={calc_result}]")
            print("AI:", answer, "\n")

            log.write(f"You: {user}\n")
            log.write(f"[route={route} | calc_result={calc_result}]\n")
            log.write(f"AI: {answer}\n\n")

    print(f"Saved log to: {log_path}")


if __name__ == "__main__":
    main()
