import json
from pathlib import Path

HERE = Path(__file__).parent
PATH_TO_TASKS = HERE.parent / "data" / "tasks.json"

def load_tasks():
    with PATH_TO_TASKS.open("r", encoding="utf-8") as f:
        tasks = json.load(f)
    print(f"Confirm loaded {len(tasks)} tasks.")
    return tasks