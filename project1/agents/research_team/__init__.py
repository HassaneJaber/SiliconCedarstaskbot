# project1/agents/research_team/__init__.py

import json
import os
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from project1.core.llm import get_llm
from project1.core.state import SupervisorState
from project1.core.text_utils import extract_json_object


RESEARCHER_SYSTEM = (
    "You are the Web Researcher Agent.\n"
    "IMPORTANT: RAG means Retrieval-Augmented Generation (retriever + context + LLM).\n"
    "You may or may not have a web-search tool enabled by the application.\n"
    "You must NOT claim you fetched sources yourself.\n"
    "Do NOT paste URLs in your JSON notes.\n"
    "\n"
    "Return JSON ONLY with this schema:\n"
    "{\n"
    "  \"topic\": str,\n"
    "  \"research_plan\": [str],\n"
    "  \"search_queries\": [str],\n"
    "  \"retrieval_metrics\": [str],\n"
    "  \"generation_metrics\": [str],\n"
    "  \"end_to_end_eval\": [str],\n"
    "  \"datasets_to_check\": [str],\n"
    "  \"baselines\": [str],\n"
    "  \"failure_modes\": [str],\n"
    "  \"sources_to_verify\": [str]\n"
    "}\n"
    "\n"
    "sources_to_verify should be 'names of trustworthy sources' (e.g., official docs, well-known tools)\n"
    "NOT made-up paper titles.\n"
)

WRITER_SYSTEM = (
    "You are the Report Writer Agent.\n"
    "Write a clean, practical report using ONLY the provided JSON notes.\n"
    "If the notes include a 'sources' list (from a web tool), you may cite and link ONLY those sources.\n"
    "If sources are missing/empty, do NOT claim web access and do NOT invent citations/URLs.\n"
    "Structure:\n"
    "- Meaning of RAG evaluation\n"
    "- Retrieval metrics\n"
    "- Generation metrics\n"
    "- End-to-end evaluation\n"
    "- Datasets/benchmarks to check\n"
    "- Baselines\n"
    "- Failure modes + tests\n"
    "- Sources (links)\n"
    "- Next actions\n"
)


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _web_provider() -> str:
    """Returns configured provider: tavily|duckduckgo|auto (or 'none/off')."""
    return (os.getenv("WEB_SEARCH_PROVIDER") or "auto").strip().lower()


def _tavily_key() -> str:
    return (os.getenv("TAVILY_API_KEY") or "").strip()


def _dedupe_sources(items: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for it in items:
        url = (it.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "title": (it.get("title") or "").strip() or url,
                "url": url,
                "snippet": (it.get("snippet") or "").strip(),
            }
        )
        if len(out) >= limit:
            break
    return out


def _search_tavily(query: str, max_results: int) -> List[Dict[str, str]]:
    """
    Tavily search. Raises on failures so caller can fallback.
    Uses a compatibility call pattern to avoid breaking if tavily-python changes args.
    """
    key = _tavily_key()
    if not key:
        raise RuntimeError("TAVILY_API_KEY missing")

    # Lazy import so project runs even if tavily isn't installed.
    from tavily import TavilyClient  # type: ignore

    client = TavilyClient(api_key=key)

    # Compatibility: try the richer signature, fall back to minimal signature.
    try:
        res = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
        )
    except TypeError:
        # Older/newer versions may not support include_* flags
        res = client.search(query=query, max_results=max_results, search_depth="basic")

    results = res.get("results") or []
    out: List[Dict[str, str]] = []
    for r in results:
        url = _safe_str(r.get("url")).strip()
        if not url:
            continue
        out.append(
            {
                "title": _safe_str(r.get("title")).strip(),
                "url": url,
                "snippet": (_safe_str(r.get("content")) or _safe_str(r.get("snippet"))).strip(),
            }
        )
    return out


def _search_ddg(query: str, max_results: int) -> List[Dict[str, str]]:
    """DuckDuckGo search via `ddgs` (successor to duckduckgo_search)."""
    # Lazy import so it's optional.
    from ddgs import DDGS  # type: ignore

    out: List[Dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = _safe_str(r.get("href")).strip()
            if not url:
                continue
            out.append(
                {
                    "title": _safe_str(r.get("title")).strip(),
                    "url": url,
                    "snippet": _safe_str(r.get("body")).strip(),
                }
            )
    return out


def _run_web_search(
    queries: List[str], per_query: int = 4, total_limit: int = 8
) -> Dict[str, Any]:
    """
    Runs web search (if configured) and returns:
      { "tool_status": {...}, "sources": [...] }

    Guarantees: sources are ONLY from the tool results (no hallucinated citations).
    """
    provider = _web_provider()

    if provider in {"none", "off", "false", "0"}:
        return {
            "tool_status": {"enabled": False, "provider": "none", "error": "disabled", "results": 0},
            "sources": [],
        }

    qlist = [q.strip() for q in (queries or []) if (q or "").strip()]
    if not qlist:
        return {
            "tool_status": {"enabled": False, "provider": "none", "error": "no queries", "results": 0},
            "sources": [],
        }

    chosen = provider
    if provider == "auto":
        chosen = "tavily" if _tavily_key() else "duckduckgo"

    sources: List[Dict[str, str]] = []
    err: str = ""

    def run_with(p: str) -> List[Dict[str, str]]:
        merged: List[Dict[str, str]] = []
        for q in qlist[:3]:  # keep it small + fast
            if p == "tavily":
                merged.extend(_search_tavily(q, per_query))
            elif p == "duckduckgo":
                merged.extend(_search_ddg(q, per_query))
            else:
                raise RuntimeError(f"Unknown provider: {p}")
        return merged

    try:
        sources = run_with(chosen)
    except Exception as e:
        err = f"{type(e).__name__}: {e}".strip()
        if provider == "auto" and chosen != "duckduckgo":
            try:
                sources = run_with("duckduckgo")
                chosen = "duckduckgo"
                err = f"tavily failed; fell back to duckduckgo ({err})"
            except Exception as e2:
                err = f"tavily failed ({err}); duckduckgo failed ({type(e2).__name__}: {e2})"
                sources = []
        else:
            sources = []

    deduped = _dedupe_sources(sources, total_limit)
    return {
        "tool_status": {
            "enabled": True,
            "provider": chosen,
            "error": err,
            "results": len(deduped),
        },
        "sources": deduped,
    }


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    return [str(x)] if str(x).strip() else []


def _normalize_notes(obj: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    notes = {
        "topic": str(obj.get("topic") or user_text).strip(),
        "research_plan": _as_list(obj.get("research_plan")),
        "search_queries": _as_list(obj.get("search_queries")),
        "retrieval_metrics": _as_list(obj.get("retrieval_metrics")),
        "generation_metrics": _as_list(obj.get("generation_metrics")),
        "end_to_end_eval": _as_list(obj.get("end_to_end_eval")),
        "datasets_to_check": _as_list(obj.get("datasets_to_check")),
        "baselines": _as_list(obj.get("baselines")),
        "failure_modes": _as_list(obj.get("failure_modes")),
        "sources_to_verify": _as_list(obj.get("sources_to_verify")),
    }
    return notes


def researcher_node(state: SupervisorState) -> dict:
    llm = get_llm(temperature=0.2)
    user_msg = state["messages"][-1].content

    out = llm.invoke(
        [
            SystemMessage(content=RESEARCHER_SYSTEM),
            HumanMessage(content=user_msg),
        ]
    )

    obj = extract_json_object(out.content) or {}
    notes = _normalize_notes(obj, user_msg)

    # Run the web tool OUTSIDE the LLM to guarantee:
    # - links are real
    # - no hallucinated citations
    web = _run_web_search(notes.get("search_queries", []))
    notes["web_search"] = web.get("tool_status")
    notes["sources"] = web.get("sources")

    msg = AIMessage(content="[RESEARCH_JSON]\n" + json.dumps(notes, indent=2))
    return {"messages": [msg], "research_notes": notes, "last_agent": "researcher", "error": ""}


def writer_node(state: SupervisorState) -> dict:
    llm = get_llm(temperature=0.2)

    notes = state.get("research_notes")
    if not notes:
        for m in reversed(state.get("messages", [])):
            if isinstance(m, AIMessage) and (m.content or "").startswith("[RESEARCH_JSON]"):
                obj = extract_json_object(m.content) or {}
                notes = obj if isinstance(obj, dict) else None
                break

    notes = notes or {"topic": "RAG evaluation", "sources_to_verify": []}

    # Ensure keys exist so the writer prompt can be strict.
    notes.setdefault("web_search", {"enabled": False, "provider": "none", "error": "", "results": 0})
    notes.setdefault("sources", [])

    out = llm.invoke(
        [
            SystemMessage(content=WRITER_SYSTEM),
            HumanMessage(content="RESEARCH_NOTES_JSON:\n" + json.dumps(notes, indent=2)),
        ]
    )

    return {"messages": [AIMessage(content=out.content)], "last_agent": "research_writer", "error": ""}


def build_research_team_graph():
    g = StateGraph(SupervisorState)
    g.add_node("researcher", researcher_node)
    g.add_node("writer", writer_node)
    g.add_edge(START, "researcher")
    g.add_edge("researcher", "writer")
    g.add_edge("writer", END)
    return g.compile()


# Export expected by supervisor:
research_app = build_research_team_graph()