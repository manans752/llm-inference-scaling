import json

import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pathlib import Path

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

MODEL_NAME = "microsoft/phi-2"
TOKEN_BUDGETS = [50,100,200]

def prompt_explicit_cot(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Answer: Let's solve step by step.\n"
    )

def prompt_constrained_cot(question: str) -> str:
    return (
        f"Question: {question.strip()}\n"
        "Answer: Solve step by step using at most 5 short steps.\n"
    )

PROMPTS = {
    "cot": prompt_explicit_cot,
    "cot_5": prompt_constrained_cot
}

HERE = Path(__file__).parent
PATH_TO_TASKS = HERE.parent / "data" / "tasks.json"
RESULTS_PATH = HERE.parent / "results" / "cot_results.csv"

with PATH_TO_TASKS.open("r", encoding="utf-8") as f:
    tasks = json.load(f)

print(f"Confirm loaded {len(tasks)} tasks.")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

config = AutoConfig.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
config.pad_token_id = config.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    dtype=torch.float32,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

model.eval()

# #returns a dictionary
# def tokenize_prompt(prompt: str):
#     return tokenizer(
#         prompt,
#         return_tensors="pt"
#     )

def generate(prompt: str, max_new_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt" )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )


    print(f"Full output shape: {outputs.shape}")
    print(f"Full output: {outputs}")

    # Slice off the prompt tokens b/c model is causal LM, it returns prompt tokens + generated tokens
    generated_tokens = outputs[0][input_len:]

    generated_list = generated_tokens.tolist()
    if 0 in generated_list:
        stop_idx = generated_list.index(0)
        generated_tokens = generated_tokens[:stop_idx]

    print(outputs)
    # Debug: print token IDs
    print(f"Generated token IDs: {generated_tokens[:10]}")  # First 10

    decoded = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )
    print(f"Decoded tokens: {decoded}")

    return decoded.strip()


results = []

for task in tasks:
    q = task["question"]
    ans = str(task["answer"])

    for strategy_name, prompt_fn in PROMPTS.items():
        for budget in TOKEN_BUDGETS:

            prompt = prompt_fn(q)
            output = generate(prompt, budget)

            correct = ans in output
            results.append({
                "id": task["id"],
                "strategy": strategy_name,
                "budget": budget,
                "output": output,
                "correct": correct
            })

            print(f"[{strategy_name}] budget={budget} correct={correct}")


PATH_TO_RESULTS = HERE.parent / "results" / "cot_results.json"

with PATH_TO_RESULTS.open("w") as f:
    json.dump(results, f, indent=2)

print("Baseline run complete.")

df = pd.DataFrame(results)
df.to_csv(RESULTS_PATH, index=False)

print("Saved results to", RESULTS_PATH)

print("\nAccuracy summary:")
print(df.groupby(["strategy", "budget"])["correct"].mean())


