import json
import torch
import pandas as pd
import re
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from .common import extract_final_answer, normalise, answers_match, is_valid_answer, majority_vote
from .model_setup import load_tokenizer, load_model
from .data_loader import load_tasks
from .generate import generate
from .results_saver import save_results

N_SAMPLES = [1,5,10]

tasks = load_tasks()
tokenizer = load_tokenizer()
model = load_model(tokenizer)

results = []
# tasks_small = tasks[1:3]
# # print(tasks_small)
for task in tasks:
    q = task["question"]
    answer = normalise(str(task["answer"]))

    prompt = (
        f"Problem: {q}\n"
        "Solve step by step.\n"
        "End with exactly:\n"
        "Final Answer: <answer>\n\n"
        "Final Answer:"
    )

    for n in N_SAMPLES:
        raw_samples = [generate(model, tokenizer, prompt, max_new_tokens=100, do_sample=True, temperature=0.5, top_p=0.95, repetition_penalty=1.1) for _ in range(n)]

        extracted = []
        for r in raw_samples:
            ans = extract_final_answer(r)
            ans = normalise(ans)

            if is_valid_answer(ans):
                extracted.append(ans)

        final_answer = majority_vote(extracted)

        # extracted = [normalise_string(extract_final_answer(s)) for s in raw_samples]
        # final_answer = majority_vote(extracted)
        correct = answers_match(final_answer, answer)

        #disagreement across extracted answers
        variance = 0
        if (len(extracted)!=0):
            variance = len(set(extracted)) / len(extracted)

        results.append({
            "id": task["id"],
            "N": n,
            "final_answer": final_answer,
            "correct": correct,
            "variance": variance
        })

        print(f"\nTask={task['id']} N={n}")
        print("Extracted:", extracted)
        print("Final vote:", final_answer)
        print("Correct:", correct)
        print("Variance:", variance)

save_results(results, "self_consistency_results.json", "self_consistency_results.csv")

print("\nAccuracy by sample count:")
df = pd.DataFrame(results)
print(df.groupby("N")["correct"].mean())