from dotenv import load_dotenv
load_dotenv()  # ensures .env is loaded even when running `python -c ...`

import os
from langchain_groq import ChatGroq


def get_llm(temperature: float = 0.0) -> ChatGroq:
    # Make sure key is actually present (not spaces)
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is missing or empty. Put it in .env and restart the terminal."
        )

    # Ensure Groq client sees it
    os.environ["GROQ_API_KEY"] = key

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=temperature,
    )