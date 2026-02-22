from dotenv import load_dotenv
load_dotenv()

import json
import os
from datetime import datetime
from langsmith import Client

DATASET_INFO_PATH = os.path.join("logs", "ex4_dataset.json")

def main():
    client = Client()

    dataset_name = f"exercise4_tools_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset = client.create_dataset(dataset_name=dataset_name)

    os.makedirs("logs", exist_ok=True)

    # ✅ Convert UUID to string so JSON can serialize it
    with open(DATASET_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"dataset_name": dataset_name, "dataset_id": str(dataset.id)},
            f,
            indent=2
        )

    print("Created dataset:", dataset_name)
    print("Dataset id:", dataset.id)
    print("Saved:", DATASET_INFO_PATH)

if __name__ == "__main__":
    main()