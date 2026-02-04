import json
import torch
import pandas as pd
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

MODEL_NAME = "microsoft/phi-2"

N_SAMPLES = [1,5,10]

HERE = Path(__file__).parent
PATH_TO_TASKS = HERE.parent / "data" / "tasks.json"
RESULTS_PATH = HERE.parent / "results" / "self_consistency_results.csv"

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

def sample(prompt: str, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, #don't always pick the most likely token, sample from distribution
            temperature=0.7, #randomness of output, will produce different reasoning paths
            top_p=0.9 #guarantees we only pick from 'reasonable' tokens
        )

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

def majority_vote(outputs):
    answers = [o.split()[-1] for o in outputs if len(o.split()) > 0]
    return Counter(answers).most_common(1)[0][0]

results = []
for task in tasks:
    q = task["question"]
    answer = str(task["answer"])

    prompt = f"Question: {q}\nAnswer:"

    for n in N_SAMPLES:

        samples = [sample(prompt) for _ in range(n)]
        final = majority_vote(samples)

        correct = answer in final

        variance = len(set(samples)) / n

        results.append({
            "id": task["id"],
            "N": n,
            "final_answer": final,
            "correct": correct,
            "variance": variance
        })

        print(f"N={n} correct={correct} variance={variance:.2f}")

PATH_TO_RESULTS = HERE.parent / "results" / "self_consistency_results.json"

with PATH_TO_RESULTS.open("w") as f:
    json.dump(results, f, indent=2)

df = pd.DataFrame(results)
df.to_csv(RESULTS_PATH, index=False)

print("Saved to", RESULTS_PATH)
print(df.groupby("N")["correct"].mean())