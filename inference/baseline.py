import os
import requests
import json
from pathlib import Path
from transformers import AutoTokenizer

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

# wraps a user question in a specific template for Mistral model inference
def format_prompt(question: str) -> str:
    return f"<s>[INST] {question} [/INST]"

def call_hf_api(prompt: str, max_new_tokens: int = 50) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens":max_new_tokens, "do_sample": False}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"API request failed with status code {response.status_code}")
    result = response.json()
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    else:
        raise ValueError(f"Unexpected API response: {result}")

def extract_answer(text: str):
    numbers = re.findall(r"\d+", text)
    return int(numbers[-1]) if numbers else None