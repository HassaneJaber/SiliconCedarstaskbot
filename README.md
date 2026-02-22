# Agentic AI Onboarding (LangChain + LangGraph) — Exercises + Project 1

This repo contains:
- **Step 4 Exercises (1–4)** from the onboarding roadmap
- **Project 1:** a multi-agent system with a **Supervisor graph** and a **Chainlit UI**

> ✅ Note: Web search is **optional** (Tavily / DuckDuckGo) and controlled via environment variables.

---

## Project Structure (high level)

- `exercise1_chain.py` — simple LLM chain
- `exercise2_rag.py` — simple RAG (TF-IDF retrieval over docs)
- `exercise3_langgraph.py` — multi-node LangGraph with branching + memory
- `exercise4_tools_agent_for_eval.py` — tool-using agent (calculator + KB retrieve + SQLite lookup)
- `ex4_make_dataset.py` / `ex4_add_examples.py` / `ex4_run_eval.py` — LangSmith dataset + eval scripts

### Project 1
- `project1/graphs/supervisor_graph.py` — main supervisor orchestrator
- `project1/agents/...` — specialized agents (SQL / research / viz / convo / RAG)
- `project1/ui/app_chainlit.py` — Chainlit UI entrypoint
- `docs/` — local knowledge base + (optional) SQLite store

---

## Requirements

- Python 3.10+ recommended
- (Optional) Conda environment

Install dependencies:

```bash
pip install -r requirements.txt