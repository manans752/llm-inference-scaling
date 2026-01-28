import os
import requests
import json
from pathlib import Path

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = os.getenv("API_URL")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

HERE = Path(__file__).parent
PATH_TO_TASKS = HERE.parent / "data" / "tasks.json"

with PATH_TO_TASKS.open("r", encoding="utf-8") as f:
    tasks = json.load(f)

print(f"Confirm loaded {len(tasks)} tasks.")

def query(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    return response.json()


def format_prompt(question: str) -> str:
    return f"<s>[INST] {question} [/INST]"