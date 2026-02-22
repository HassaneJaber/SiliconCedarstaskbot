from dotenv import load_dotenv
load_dotenv()

import json
import os
from langsmith import Client

DATASET_INFO_PATH = os.path.join("logs", "ex4_dataset.json")

def get_dataset_name() -> str:
    # Prefer env var if you want to point to a specific dataset
    env_name = os.getenv("DATASET_NAME")
    if env_name:
        return env_name

    # Else read the last created dataset from logs/ex4_dataset.json
    if os.path.exists(DATASET_INFO_PATH):
        with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)["dataset_name"]

    raise SystemExit(
        "Can't find dataset name.\n"
        "Either set DATASET_NAME in your .env, or run ex4_make_dataset.py (and ensure it saves logs/ex4_dataset.json)."
    )

def main():
    client = Client()
    dataset_name = get_dataset_name()

    # 12 small examples: math, KB questions, and “not in context but similar terms”
    inputs = [
        {"question": "13*13+1"},
        {"question": "calc: (8*12)+12"},
        {"question": "7*(5+3)"},
        {"question": "100/4 + 6"},

        {"question": "What is LangGraph?"},
        {"question": "What is LangGraph used for?"},
        {"question": "What is the purpose of RAG?"},
        {"question": "How is memory often done in LangGraph?"},

        {"question": "What year was LangGraph released?"},
        {"question": "Who created LangGraph?"},
        {"question": "Does LangGraph support human-in-the-loop? Explain from the context."},
        {"question": "What is the default port of LangGraph?"},
    ]

    outputs = [
        {"expected_route": "calc_answer", "expected_contains": ["170"]},
        {"expected_route": "calc_answer", "expected_contains": ["108"]},
        {"expected_route": "calc_answer", "expected_contains": ["56"]},
        {"expected_route": "calc_answer", "expected_contains": ["31"]},

        {"expected_route": "rag", "expected_contains": ["orchestration", "workflows", "graphs"]},
        {"expected_route": "rag", "expected_contains": ["used", "agent workflows"]},
        {"expected_route": "rag", "expected_contains": ["retrieves", "chunks", "context"]},
        {"expected_route": "rag", "expected_contains": ["checkpoint"]},

        {"expected_route": "rag", "expected_contains": ["i don't know based on the provided context"]},
        {"expected_route": "rag", "expected_contains": ["i don't know based on the provided context"]},
        {"expected_route": "rag", "expected_contains": ["i don't know based on the provided context"]},
        {"expected_route": "rag", "expected_contains": ["i don't know based on the provided context"]},
    ]

    client.create_examples(dataset_name=dataset_name, inputs=inputs, outputs=outputs)
    print(f"Added {len(inputs)} examples to dataset: {dataset_name}")

if __name__ == "__main__":
    main()
