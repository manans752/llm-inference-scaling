import os
import pandas as pd

from inference.common.common import extract_final_answer, normalise, answers_match
from inference.common.model_setup import load_tokenizer, load_model
from inference.common.data_loader import load_tasks
from inference.common.generate import generate
from inference.common.results_saver import save_results

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

tasks = load_tasks()
tokenizer = load_tokenizer()
model = load_model(tokenizer)


def format_prompt(q: str):
    return (
        f"Problem: {q}\n"
        "Solve step by step.\n"
        "End with exactly:\n"
        "Final Answer: <answer>\n\n"
        "Final Answer:"
    )


results = []
for task in tasks:
    q = task["question"]
    answer = normalise(str(task["answer"]))

    prompt = format_prompt(q)
    output = generate(model, tokenizer, prompt, max_new_tokens=100)
    prediction = normalise(extract_final_answer(output))

    correct = answers_match(prediction, answer)

    results.append({
        "id": task["id"],
        "prediction": prediction,
        "correct": correct
    })

save_results(results, "baseline_results.json", "baseline_results.csv")

print("Accuracy:", pd.DataFrame(results)["correct"].mean())