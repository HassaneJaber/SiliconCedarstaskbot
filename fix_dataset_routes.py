from dotenv import load_dotenv
load_dotenv()

from langsmith import Client

client = Client()

DATASET_NAME = "exercise4_tools_20260219_110633"  # <-- CHANGE THIS

# get dataset
ds = client.read_dataset(dataset_name=DATASET_NAME)

# update examples
examples = list(client.list_examples(dataset_id=ds.id))

fixed = 0
for ex in examples:
    outs = ex.outputs or {}
    if outs.get("expected_route") == "rag_answer":
        outs["expected_route"] = "rag"
        client.update_example(example_id=ex.id, outputs=outs)
        fixed += 1

print(f"Updated {fixed} examples from rag_answer -> rag in dataset: {DATASET_NAME}")
