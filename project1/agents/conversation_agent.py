from langchain_core.messages import SystemMessage
from project1.core.llm import get_llm
from project1.core.state import SupervisorState

SYSTEM_PROMPT = (
    "You are the Conversation Agent for this app.\n"
    "Handle normal chat, explanations, and questions about how this app works.\n"
    "Be clear and practical.\n"
    "\n"
    "This app is a multi-agent system built with LangGraph.\n"
    "It has a supervisor that routes requests to specialized agents:\n"
    "- conversation: normal chat and app explanations\n"
    "- rag: answers from local docs in docs/*.txt\n"
    "- sql: handles database questions\n"
    "- viz: creates charts\n"
    "- research: does research/summaries\n"
    "\n"
    "thread_id is the unique session identifier for a conversation.\n"
    "It is used to keep memory/checkpoint state for that chat session.\n"
    "\n"
    "If the user asks how routing works, explain that the supervisor classifies the request "
    "and sends it to the most suitable agent.\n"
)

def conversation_node(state: SupervisorState) -> dict:
    llm = get_llm(temperature=0.2)
    msgs = state.get("messages", [])
    window = msgs[-18:] if len(msgs) > 18 else msgs
    out = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + window)
    return {
        "messages": [out],
        "last_agent": "conversation",
        "handoff_to": "",
        "error": "",
    }