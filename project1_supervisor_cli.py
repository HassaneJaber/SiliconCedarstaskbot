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
# KB loading (docs/*.txt preferred)
# ----------------------------
def load_kb_chunks() -> Tuple[List[str], List[str]]:
    """
    Returns (chunks, sources) where sources[i] is the filename for chunks[i].
    """
    kb_files = glob.glob(os.path.join("docs", "*.txt"))
    if not kb_files and os.path.exists("kb.txt"):
        kb_files = ["kb.txt"]

    if not kb_files:
        os.makedirs("docs", exist_ok=True)
        fallback_path = os.path.join("docs", "kb.txt")
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write(
                "LangGraph is a low-level orchestration framework for building long-running, stateful LLM workflows as graphs.\n"
                "You define nodes (steps) and edges (transitions), so execution can branch, loop, and resume with memory/persistence.\n"
                "It’s used to build reliable agent workflows (routing, tools, multi-step plans, human-in-the-loop, etc.).\n"
                "RAG retrieves relevant chunks from documents and gives them to the LLM as context.\n"
                "If the answer is not in the context, the assistant should say it doesn't know based on the provided context.\n"
                "Memory in LangGraph is often done using a checkpointer and a thread_id to persist state across turns.\n"
            )
        kb_files = [fallback_path]

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
        for c in chunk_text(text):
            chunks.append(c)
            sources.append(os.path.basename(path))

    return chunks, sources


CHUNKS, SOURCES = load_kb_chunks()
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(CHUNKS)


def retrieve_top(query: str, k: int = 3) -> Tuple[float, str, str]:
    q_vec = vectorizer.transform([query])
    sims = (X @ q_vec.T).toarray().ravel()

    if sims.size == 0:
        return 0.0, "", ""

    top_idx = sims.argsort()[-k:][::-1]
    top_score = float(sims[top_idx[0]]) if len(top_idx) else 0.0

    top_chunks = [CHUNKS[i] for i in top_idx if i < len(CHUNKS)]
    top_sources = sorted(set(SOURCES[i] for i in top_idx if i < len(SOURCES)))

    context = "\n\n---\n\n".join(top_chunks).strip()
    sources_str = ", ".join(top_sources)
    return top_score, context, sources_str


# ----------------------------
# Calculator (safe-ish)
# ----------------------------
MATH_ALLOWED = re.compile(r"^[0-9\.\+\-\*\/\(\)\s]+$")

def looks_like_math(s: str) -> bool:
    s = s.strip().lower()
    if s.startswith("calc:"):
        return True
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
    sources: str
    calc_result: str


# ----------------------------
# Nodes (Supervisor + 3 agents)
# ----------------------------
def supervisor_router(state: State) -> dict:
    user_text = state["messages"][-1].content

    # reset per-turn fields (prevents stale prints)
    base = {"context": "", "score": 0.0, "sources": "", "calc_result": ""}

    if looks_like_math(user_text):
        return {**base, "route": "math_agent"}

    score, context, sources = retrieve_top(user_text, k=3)
    if score >= THRESHOLD and context:
        return {**base, "route": "rag_agent", "score": score, "context": context, "sources": sources}

    return {**base, "route": "chat_agent"}


def math_agent(state: State) -> dict:
    user_text = state["messages"][-1].content
    expr = user_text.split("calc:", 1)[1] if user_text.lower().startswith("calc:") else user_text
    calc_result = safe_calc(expr)

    sys = SystemMessage(content="You are a math assistant. Use the calculator result and answer clearly.")
    msg = HumanMessage(content=f"User input: {user_text}\nCalculator result: {calc_result}\nGive the final answer.")
    resp = llm.invoke([sys, msg])

    return {"messages": [resp], "calc_result": calc_result, "route": "math_agent"}


def rag_agent(state: State) -> dict:
    context = state.get("context", "")
    sources = state.get("sources", "")
    question = state["messages"][-1].content

    sys = SystemMessage(content=(
        "You are a helpful assistant.\n"
        "Answer using ONLY the provided CONTEXT.\n"
        "If the answer is not in the context, say: \"I don't know based on the provided context.\""
    ))
    msg = HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAdd a last line exactly like:\nSources: [{sources}]")
    resp = llm.invoke([sys, msg])

    return {"messages": [resp], "route": "rag_agent"}


def chat_agent(state: State) -> dict:
    sys = SystemMessage(content=(
        "You are a helpful conversational assistant.\n"
        "Use prior messages if available. Be accurate.\n"
        "If unsure, say you are unsure instead of guessing."
    ))
    resp = llm.invoke([sys] + state["messages"])
    return {"messages": [resp], "route": "chat_agent"}


def route_fn(state: State) -> str:
    return state["route"]


# ----------------------------
# Build Graph
# ----------------------------
builder = StateGraph(State)
builder.add_node("supervisor", supervisor_router)
builder.add_node("math_agent", math_agent)
builder.add_node("rag_agent", rag_agent)
builder.add_node("chat_agent", chat_agent)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_fn, {
    "math_agent": "math_agent",
    "rag_agent": "rag_agent",
    "chat_agent": "chat_agent",
})

builder.add_edge("math_agent", END)
builder.add_edge("rag_agent", END)
builder.add_edge("chat_agent", END)


def main():
    thread_id = "project1-thread-1"  # change this to start a fresh memory thread
    print("\nProject 1 (MVP): Supervisor + 3 Agents (Math + RAG + Chat) with SQLite persistence")
    print(f"Threshold={THRESHOLD} | thread_id={thread_id}")
    print("Type 'exit' to stop.\n")

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"project1_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # IMPORTANT: from_conn_string is a context manager in your version
    with SqliteSaver.from_conn_string("project1_checkpoints.db") as checkpointer:
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
                sources = result.get("sources", "")
                calc_result = result.get("calc_result", "")

                print(f"\n[route={route} | score={score:.3f} | calc_result={calc_result} | sources={sources}]")
                print("AI:", answer, "\n")

                log.write(f"You: {user}\n")
                log.write(f"[route={route} | score={score:.3f} | calc_result={calc_result} | sources={sources}]\n")
                log.write(f"AI: {answer}\n\n")

    print(f"Saved log to: {log_path}")


if __name__ == "__main__":
    main()
