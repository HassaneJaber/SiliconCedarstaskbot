from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
import uuid
from langsmith import Client

from exercise4_tools_agent_for_eval import run_agent  # <-- must exist

DATASET_INFO_PATH = os.path.join("logs", "ex4_dataset.json")

def get_dataset_name() -> str:
    env_name = os.getenv("DATASET_NAME")
    if env_name:
        return env_name
    if os.path.exists(DATASET_INFO_PATH):
        with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)["dataset_name"]
    raise SystemExit(
        "Can't find dataset name.\n"
        "Set DATASET_NAME in .env or run ex4_make_dataset.py (saving logs/ex4_dataset.json)."
    )

def predict(inputs: dict) -> dict:
    q = inputs["question"]
    # New thread id per example so state doesn’t leak between eval rows
    thread_id = f"ex4-eval-{uuid.uuid4().hex[:8]}"
    return run_agent(q, thread_id=thread_id)

def route_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    expected = (reference_outputs.get("expected_route") or "").strip()
    got = (outputs.get("route") or "").strip()
    return {"score": float(expected == got)}

def content_contains(outputs: dict, reference_outputs: dict) -> dict:
    expected_list = reference_outputs.get("expected_contains") or []
    if isinstance(expected_list, str):
        expected_list = [expected_list]

    answer = (outputs.get("response") or "").lower()
    ok = True
    missing = []
    for s in expected_list:
        s2 = str(s).lower()
        if s2 not in answer:
            ok = False
            missing.append(s2)

    # For calc outputs, also accept numeric match even if phrasing differs
    if not ok and expected_list and any(x.strip().isdigit() for x in map(str, expected_list)):
        nums = re.findall(r"-?\d+(?:\.\d+)?", answer)
        for x in expected_list:
            x = str(x).strip()
            if x.isdigit() and x in nums:
                return {"score": 1.0}

    return {"score": float(ok)}

def main():
    client = Client()
    dataset_name = get_dataset_name()

    results = client.evaluate(
        predict,
        data=dataset_name,
        evaluators=[route_accuracy, content_contains],
        experiment_prefix="Exercise4-Tools-Eval",
        description="Evaluate tool routing + response correctness for Exercise 4.",
    )

    print("\nDone. Check LangSmith → Datasets →", dataset_name, "→ Experiments.")
    return results

if __name__ == "__main__":
    main()
