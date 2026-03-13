import os
import pandas as pd

from .common import extract_final_answer, normalise, answers_match
from .model_setup import load_tokenizer, load_model
from .data_loader import load_tasks
from .generate import generate
from .results_saver import save_results

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

TOKEN_BUDGETS = [50, 100, 200]

def prompt_explicit_cot(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Solve step by step.\n"
        "End with: Final Answer: <answer>\n"
    )

def prompt_constrained_cot(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Solve in at most 5 short steps.\n"
        "End with: Final Answer: <answer>\n"
    )

PROMPTS = {
    "cot": prompt_explicit_cot,
    "cot_5": prompt_constrained_cot
}

tasks = load_tasks()
tokenizer = load_tokenizer()
model = load_model(tokenizer)

results = []
for task in tasks:
    q = task["question"]
    ans = normalise(str(task["answer"]))

    for strategy_name, prompt_fn in PROMPTS.items():
        for budget in TOKEN_BUDGETS:
            prompt = prompt_fn(q)
            output = generate(model, tokenizer, prompt, max_new_tokens=budget)
            prediction = normalise(extract_final_answer(output))
            correct = answers_match(prediction, ans)

            results.append({
                "id": task["id"],
                "strategy": strategy_name,
                "tokens_used": budget,
                "output": output,
                "correct": correct
            })

            print(f"[{strategy_name}] budget={budget} correct={correct}")

save_results(results, "cot_results.json", "cot_results.csv")

print("\nAccuracy by strategy and budget:")
df = pd.DataFrame(results)
print(df.groupby(["strategy", "tokens_used"])["correct"].mean())