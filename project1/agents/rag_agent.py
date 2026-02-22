import glob
import os
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_core.messages import AIMessage, SystemMessage

from project1.core.llm import get_llm
from project1.core.state import SupervisorState

RAG_SYSTEM = (
    "You are the RAG Agent.\n"
    "Answer using ONLY the provided context.\n"
    "If the answer is not in the context, say exactly:\n"
    "I don't know based on the provided context."
)

BASE_DIR = Path(__file__).resolve().parents[2]  # .../agentic-ai
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "docs")))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
THRESHOLD = float(os.getenv("RAG_THRESHOLD", "0.12"))

class TfidfStore:
    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.texts: List[str] = []
        self.sources: List[str] = []
        self.X = None

    def build(self):
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(glob.glob(str(self.docs_dir / "*.txt")))
        if not files:
            # create fallback kb.txt
            fallback = self.docs_dir / "kb.txt"
            fallback.write_text(
                "LangGraph is a framework for building stateful LLM workflows as graphs.\n"
                "RAG retrieves relevant chunks from documents and provides them as context.\n",
                encoding="utf-8",
            )
            files = [str(fallback)]

        texts, sources = [], []
        for p in files:
            t = Path(p).read_text(encoding="utf-8", errors="ignore").strip()
            if t:
                texts.append(t)
                sources.append(Path(p).name)

        self.texts = texts
        self.sources = sources
        self.X = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, k: int) -> Tuple[str, float, str]:
        if not self.texts or self.X is None:
            return "", 0.0, ""
        qv = self.vectorizer.transform([query])
        sims = (self.X @ qv.T).toarray().ravel()
        top_idx = sims.argsort()[-k:][::-1]
        top_score = float(sims[top_idx[0]]) if len(top_idx) else 0.0

        ctx_parts = [self.texts[i] for i in top_idx if i < len(self.texts)]
        srcs = sorted(set(self.sources[i] for i in top_idx if i < len(self.sources)))
        return "\n\n---\n\n".join(ctx_parts), top_score, ", ".join(srcs)

_STORE = TfidfStore(DOCS_DIR)
_STORE.build()

def rag_node(state: SupervisorState) -> dict:
    llm = get_llm(temperature=0.0)
    user_msg = state["messages"][-1].content.strip()
    low = user_msg.lower()

    # ✅ Deterministic "list docs" behavior
    if any(x in low for x in ["what docs", "list docs", "what is in my docs", "show docs", "which documents", "files in docs"]):
        try:
            names = sorted([p.name for p in DOCS_DIR.glob("*.txt")])
            if not names:
                answer = "docs/ contains no .txt files."
            else:
                answer = "docs/ contains these .txt files:\n- " + "\n- ".join(names)
            return {
                "messages": [AIMessage(content=answer)],
                "last_agent": "rag",
                "rag_sources": ", ".join(names),
                "error": "",
            }
        except Exception:
            return {
                "messages": [AIMessage(content="I couldn't read the docs folder.")],
                "last_agent": "rag",
                "rag_sources": "",
                "error": "",
            }

    # Normal RAG
    context, score, sources = _STORE.retrieve(user_msg, k=TOP_K)

    if (not context.strip()) or (score < THRESHOLD):
        return {
            "messages": [AIMessage(content="I don't know based on the provided context.")],
            "last_agent": "rag",
            "rag_sources": sources,
            "error": "",
        }

    out = llm.invoke([
        SystemMessage(content=RAG_SYSTEM),
        SystemMessage(content=f"CONTEXT:\n{context}")
    ] + state.get("messages", [])[-6:])

    content = (out.content or "").strip()
    if not content:
        content = "I don't know based on the provided context."

    return {"messages": [AIMessage(content=content)], "last_agent": "rag", "rag_sources": sources, "error": ""}