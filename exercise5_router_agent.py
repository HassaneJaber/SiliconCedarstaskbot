from dotenv import load_dotenv
load_dotenv()

import os
import re
import glob
from datetime import datetime
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict

from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


# ----------------------------
# Settings
# ----------------------------
MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
THRESHOLD = 0.12  # retrieval confidence threshold

llm = ChatGroq(model=MODEL, temperature=0)


# ----------------------------
# Load KB (prefer docs/*.txt, else kb.txt, else create kb.txt)
# ----------------------------
def load_kb_chunks() -> Tuple[List[str], List[str]]:
    """
    Returns (chunks, sources) where sources[i] is the relative file path for chunks[i].
    """
    kb_files = glob.glob(os.path.join("docs", "*.txt"))
    if not kb_files and os.path.exists("kb.txt"):
        kb_files = ["kb.txt"]

    if not kb_files:
        # create a fallback kb.txt so the script always runs
        with open("kb.txt", "w", encoding="utf-8") as f:
            f.write(
                "LangChain is a framework for building LLM applications using prompts, models, chains, and tools.\n"
                "LangGraph is a low-level orchestration framework for long-running, stateful workflows/agents, with branching and loops.\n"
                "RAG (retrieval-augmented generation) retrieves relevant chunks from documents and gives them to the LLM as context.\n"
                "If the answer is not in the context, the assistant should say it doesn't know based on the provided context.\n"
            )
        kb_files = ["kb.txt"]

    def chunk_text(text: str, chunk_words: int = 90, overlap_words: int = 25) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i + chunk_words]
            chunks.append(" ".join(chunk))
            i += max(1, chunk_words - overlap_words)
        return [c.strip() for c in chunks if c.strip()]

    chunks, sources = [], []
    for path in kb_files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        rel = os.path.relpath(path)  # shows docs\kb.txt instead of just kb.txt
        for c in chunk_text(text):
            chunks.append(c)
            sources.append(rel)

    return chunks, sources


CHUNKS, SOURCES = load_kb_chunks()
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(CHUNKS)


# ----------------------------
# Calculator tool (safe-ish)
# ----------------------------
MATH_ALLOWED = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]+$")

def looks_like_math(s: str) -> bool:
    s = s.strip().lower()
    if s.startswith("calc:"):
        return True
    # heuristic: contains digits and an operator
    return bool(re.search(r"\d", s) and re.search(r"[\+\-\*\/\(\)]", s))

def safe_calc(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return "Error: empty expression."
    if not MATH_ALLOWED.match(expr):
        return "Error: expression contains invalid characters."
    try:
        val = eval(expr, {"__builtins__": {}}, {})
        return str(val)
    except Exception as e:
        return f"Error: {e}"


# ----------------------------
# LangGraph State
# ----------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    context: str
    score: float
    source_files: str
    calc_result: str


# ----------------------------
# Nodes
# ----------------------------
def router_node(state: State) -> dict:
    """Choose calc vs retrieve."""
    user_text = state["messages"][-1].content

    # Always reset these each turn (prevents stale display)
    base = {"calc_result": "", "context": "", "score": 0.0, "source_files": ""}

    if looks_like_math(user_text):
        return {**base, "route": "calc"}

    return {**base, "route": "retrieve"}


def retrieve_node(state: State) -> dict:
    """TF-IDF retrieve top chunk(s)."""
    query = state["messages"][-1].content
    q_vec = vectorizer.transform([query])
    sims = (X @ q_vec.T).toarray().ravel()

    top_idx = sims.argsort()[-3:][::-1]
    top_score = float(sims[top_idx[0]]) if len(top_idx) else 0.0

    top_chunks = [CHUNKS[i] for i in top_idx if i < len(CHUNKS)]
    top_sources = sorted(set(SOURCES[i] for i in top_idx if i < len(SOURCES)))

    context = "\n\n---\n\n".join(top_chunks)
    route = "rag" if top_score >= THRESHOLD and context.strip() else "direct"

    return {
        "route": route,
        "context": context,
        "score": top_score,
        "source_files": ", ".join(top_sources)
    }


def calc_node(state: State) -> dict:
    user_text = state["messages"][-1].content
    expr = user_text.split("calc:", 1)[1] if "calc:" in user_text.lower() else user_text
    return {"calc_result": safe_calc(expr), "route": "calc_answer"}


def rag_answer_node(state: State) -> dict:
    context = state.get("context", "")
    sources = state.get("source_files", "")

    sys = SystemMessage(content=(
        "You are a helpful assistant.\n"
        "Answer using ONLY the provided CONTEXT.\n"
        "If the answer is not in the context, say: \"I don't know based on the provided context.\""
    ))
    msg = HumanMessage(
        content=(
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{state['messages'][-1].content}\n\n"
            f"Add a last line: Sources: [{sources}]"
        )
    )
    resp = llm.invoke([sys, msg])
    return {"messages": [resp]}


def direct_answer_node(state: State) -> dict:
    sys = SystemMessage(content=(
        "You are a helpful assistant. Be accurate.\n"
        "If you are unsure, say you are unsure instead of guessing."
    ))
    resp = llm.invoke([sys] + state["messages"])
    return {"messages": [resp]}


def calc_answer_node(state: State) -> dict:
    sys = SystemMessage(content="You are a helpful assistant. Use the calculator result to answer clearly.")
    msg = HumanMessage(
        content=(
            f"User: {state['messages'][-1].content}\n"
            f"Calculator result: {state.get('calc_result','')}\n"
            "Give the final answer."
        )
    )
    resp = llm.invoke([sys, msg])
    return {"messages": [resp]}


def route_fn(state: State) -> str:
    return state["route"]


# ----------------------------
# Build graph (builder only — no compile here!)
# ----------------------------
builder = StateGraph(State)

builder.add_node("router", router_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("rag", rag_answer_node)
builder.add_node("direct", direct_answer_node)
builder.add_node("calc", calc_node)
builder.add_node("calc_answer", calc_answer_node)

builder.add_edge(START, "router")

builder.add_conditional_edges("router", route_fn, {
    "calc": "calc",
    "retrieve": "retrieve",
})

builder.add_conditional_edges("retrieve", route_fn, {
    "rag": "rag",
    "direct": "direct",
})

builder.add_edge("calc", "calc_answer")
builder.add_edge("calc_answer", END)
builder.add_edge("rag", END)
builder.add_edge("direct", END)


def main():
    thread_id = "router-thread-1"
    print("\nExercise 5: Router Agent (Calc + RAG + Direct) with SQLite persistence")
    print(f"Threshold={THRESHOLD} | thread_id={thread_id}")
    print("Type 'exit' to stop.\n")

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"router_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # IMPORTANT: SqliteSaver.from_conn_string returns a context manager in your version
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

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
                score = result.get("score", 0.0)
                sources = result.get("source_files", "")
                calc_result = result.get("calc_result", "")

                print(f"\n[route={route} | score={score:.3f} | calc_result={calc_result} | sources={sources}]")
                print("AI:", answer, "\n")

                log.write(f"You: {user}\n")
                log.write(f"[route={route} | score={score:.3f} | calc_result={calc_result} | sources={sources}]\n")
                log.write(f"AI: {answer}\n\n")

    print(f"Saved log to: {log_path}")


if __name__ == "__main__":
    main()
