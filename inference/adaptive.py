import json
import re
import torch
import pandas as pd
from pathlib import Path

from .common import extract_final_answer, normalise, answers_match, low_confidence
from .model_setup import load_tokenizer, load_model
from .data_loader import load_tasks
from .generate import generate
from .results_saver import save_results

tasks = load_tasks()
tokenizer = load_tokenizer()
model = load_model(tokenizer)

def base_prompt(q: str):
    return (
        "Solve the problem and output ONLY the final number.\n\n"
        f"Problem: {q}\n"
        "Final Answer:"
    )

def cot_prompt(q: str):
    return (
        "Solve the problem step by step.\n"
        "End with 'Final Answer: <number>'.\n\n"
        f"Problem: {q}\n"
    )

results = []
# tasks_small = tasks[1:3]
# print(tasks_small)

for task in tasks:
    q = task["question"]
    answer = normalise(str(task["answer"]))

    prompt = base_prompt(q)
    output = generate(model, tokenizer, prompt, max_new_tokens=20)
    prediction = normalise(extract_final_answer(output))
    tokens_used = 20

    if prediction is None or low_confidence(output):
        prompt = cot_prompt(q)
        output = generate(model, tokenizer, prompt, max_new_tokens=120)
        prediction = normalise(extract_final_answer(output))
        tokens_used = 120

    correct = answers_match(prediction, answer)
    print(f"Pred: {prediction} | True: {answer}")
    print("Correct:", correct, "Tokens:", tokens_used)

    results.append({
        "id": task["id"],
        "prediction": prediction,
        "answer": answer,
        "correct": correct,
        "tokens_used": tokens_used
    })

save_results(results, "adaptive_results.json", "adaptive_results.csv")

df = pd.DataFrame(results)
print("Accuracy:", df["correct"].mean())
print("Avg tokens:", df["tokens_used"].mean())