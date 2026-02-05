import json
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

MODEL_NAME = "microsoft/phi-2"

HERE = Path(__file__).parent
PATH_TO_TASKS = HERE.parent / "data" / "tasks.json"
RESULTS_PATH = HERE.parent / "results" / "adaptive_results.csv"

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
    device_map="cpu"
)

model.eval()

def generate(prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    # gen_tokens = output_ids[0][input_len:]
    # return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    generated_tokens = output_ids[0][input_len:]

    generated_list = generated_tokens.tolist()
    if 0 in generated_list:
        stop_idx = generated_list.index(0)
        generated_tokens = generated_tokens[:stop_idx]

    decoded = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return decoded.strip()

def low_confidence(output: str):
    if len(output.split()) <2: #too short
        return True
    if not any(ch.isdigit() for ch in output): #no number in answer
        return True
    return False

results = []
tasks_small = tasks[1:3]
print(tasks_small)

for task in tasks_small:
    q = task["question"]
    answer = str(task["answer"])

    prompt = f"Question: {q}\nAnswer: (only include the final result, dont restate the question)"
    output = generate(prompt, 30)
    tokens_used = 30

    conf_low = low_confidence(output);
    if conf_low:
        prompt = f"Question: {q}\nAnswer: Let's solve step by step.\n"
        output = generate(prompt, 150)
        tokens_used = 150

    correct = str(task["answer"]) in output

    results.append({
        "id": task["id"],
        "output": output,
        "correct": correct,
        "tokens_used": tokens_used,
        "low confidence": conf_low,
    })

    print("Correct:", correct, "Tokens:", tokens_used)


PATH_TO_RESULTS = HERE.parent / "results" / "adaptive_results.json"

with PATH_TO_RESULTS.open("w") as f:
    json.dump(results, f, indent=2)

df = pd.DataFrame(results)
df.to_csv(RESULTS_PATH, index=False)
print("Saved to", RESULTS_PATH)

print("Accuracy:", df["correct"].mean())
print("Avg tokens:", df["tokens_used"].mean())