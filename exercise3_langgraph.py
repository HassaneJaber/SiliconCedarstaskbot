import os
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from typing import Annotated
from typing_extensions import TypedDict


# ----------------------------
# 0) Setup
# ----------------------------
load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
KB_PATH = "docs/kb.txt"

# If kb.txt doesn't exist, create a small one so the exercise always runs.
# IMPORTANT: keep KB factual (no "if not in context say I don't know" rule inside KB text)
if not os.path.exists(KB_PATH):
    os.makedirs("docs", exist_ok=True)
    with open(KB_PATH, "w", encoding="utf-8") as f:
        f.write(
            "LangChain is a framework for building LLM applications using prompts, models, chains, and tools.\n"
            "LangGraph is a low-level orchestration framework for building long-running, stateful LLM workflows as graphs.\n"
            "In LangGraph you define nodes (steps) and edges (transitions); execution can branch, loop, and resume.\n"
            "RAG (retrieval-augmented generation) retrieves relevant chunks from documents and gives them to the LLM as context.\n"
            "Memory in LangGraph is often done using checkpointing (a checkpointer) and a thread_id to persist state across turns.\n"
            "LangSmith is LangChain’s platform for tracing, debugging, testing, and evaluating LLM applications.\n"
        )

with open(KB_PATH, "r", encoding="utf-8") as f:
    KB_TEXT = f.read()

# Remove “meta-instruction” lines if they exist in the KB (from older versions)
def clean_kb(text: str) -> str:
    bad_phrases = [
        "If the answer is not in the context",
        "the assistant should say it doesn't know",
        "I don't know based on the provided context",
    ]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        if any(bp.lower() in ln.lower() for bp in bad_phrases):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()

KB_TEXT = clean_kb(KB_TEXT)

# ✅ Better chunking: each line is its own chunk (clean retrieval)
CHUNKS = [ln.strip() for ln in KB_TEXT.splitlines() if ln.strip()]

# TF-IDF setup
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(CHUNKS) if CHUNKS else None

llm = ChatGroq(model=MODEL, temperature=0)

# Retrieval controls
TOP_K = 3
THRESHOLD = 0.08  # lower because line-chunks are short


# ----------------------------
# 1) State
# ----------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    top_score: float
    route: str


# ----------------------------
# 2) Nodes
# ----------------------------
def retrieve_node(state: State) -> dict:
    """Retrieve top-k lines from kb.txt using TF-IDF similarity."""
    query = state["messages"][-1].content.strip()

    if not CHUNKS or X is None:
        return {"context": "", "top_score": 0.0, "route": "direct_answer"}

    q_vec = vectorizer.transform([query])
    scores = (X @ q_vec.T).toarray().ravel()

    # Get top-k indices sorted by score desc
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    top_score = float(scores[top_idx[0]]) if len(top_idx) else 0.0

    # Build context from the best lines (only keep lines with >0 score)
    selected_lines = [CHUNKS[i] for i in top_idx if scores[i] > 0]
    context = "\n".join(f"- {ln}" for ln in selected_lines).strip()

    route = "rag_answer" if top_score >= THRESHOLD and context else "direct_answer"

    return {"context": context, "top_score": top_score, "route": route}


def rag_answer_node(state: State) -> dict:
    """Answer using retrieved context + full conversation memory."""
    context = state.get("context", "").strip()

    sys = SystemMessage(
        content=(
            "You are a helpful assistant.\n"
            "Use ONLY the retrieved context to answer.\n"
            "If the answer is not clearly stated in the context, reply exactly:\n"
            "\"I don't know based on the provided context.\""
        )
    )
    ctx = SystemMessage(content=f"Retrieved context:\n{context}")

    messages = [sys, ctx] + state["messages"]
    resp = llm.invoke(messages)
    return {"messages": [resp]}


def direct_answer_node(state: State) -> dict:
    """Answer normally (no RAG), still using memory."""
    sys = SystemMessage(content="You are a helpful assistant.")
    messages = [sys] + state["messages"]
    resp = llm.invoke(messages)
    return {"messages": [resp]}


def route_fn(state: State) -> str:
    return state["route"]


# ----------------------------
# 3) Build Graph (3 nodes + branching + memory)
# ----------------------------
builder = StateGraph(State)

builder.add_node("retrieve", retrieve_node)
builder.add_node("rag_answer", rag_answer_node)
builder.add_node("direct_answer", direct_answer_node)

builder.add_edge(START, "retrieve")
builder.add_conditional_edges("retrieve", route_fn)

builder.add_edge("rag_answer", END)
builder.add_edge("direct_answer", END)

graph = builder.compile(checkpointer=InMemorySaver())


# ----------------------------
# 4) Run loop (multi-turn memory)
# ----------------------------
def main():
    thread_id = "onboarding-thread-2"  # change this to reset memory
    print("\nExercise 3: LangGraph multi-node + branching + memory")
    print("Type 'exit' to stop.\n")

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"langgraph_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with open(log_path, "w", encoding="utf-8") as log:
        while True:
            user = input("You: ").strip()
            if user.lower() in ("exit", "quit"):
                print("Bye!")
                break

            result = graph.invoke(
                {"messages": [HumanMessage(content=user)]},
                {"configurable": {"thread_id": thread_id}},
            )

            answer = result["messages"][-1].content
            route = result.get("route", "?")
            score = result.get("top_score", 0.0)

            print(f"\n[route={route} | score={score:.3f}]")
            print("AI:", answer, "\n")

            log.write(f"You: {user}\n")
            log.write(f"[route={route} | score={score:.3f}]\n")
            log.write(f"AI: {answer}\n\n")

    print(f"Saved log to: {log_path}")


if __name__ == "__main__":
    main()