from langchain_core.messages import SystemMessage
from project1.core.llm import get_llm
from project1.core.state import SupervisorState

SYSTEM_PROMPT = (
    "You are the Conversation Agent.\n"
    "Handle normal chat and explanations.\n"
    "Be clear and practical."
)

def conversation_node(state: SupervisorState) -> dict:
    llm = get_llm(temperature=0.2)
    msgs = state.get("messages", [])
    window = msgs[-18:] if len(msgs) > 18 else msgs
    out = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + window)
    return {"messages": [out], "last_agent": "conversation", "error": ""}