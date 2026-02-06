import json
import re
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

def extract_final_answer(text: str):
    match = re.search(r"Final Answer:\s*(.*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[0] if lines else None


def normalise(ans: str): # yes = YES
    if ans is None:
        return None
    return ans.strip().lower()

def answers_match(pred: str, target_answer: str): #ensure semantically equal answers are not marked false
    if pred is None or target_answer is None:
        return False

    pred = pred.strip().lower()
    target_answer = target_answer.strip().lower()

    # target_answer  is numeric
    if target_answer.replace(".", "", 1).isdigit():

        # extract first number from prediction
        match = re.search(r"-?\d+(\.\d+)?", pred)

        if match:
            pred_num = match.group(0)
            return pred_num == target_answer

        return False

    # target_answer is text
    pred = re.sub(r"[^a-z0-9]", "", pred)
    target_answer = re.sub(r"[^a-z0-9]", "", target_answer)

    return pred == target_answer

# def low_confidence(output: str): #True if output seems unreliable
#     ans = extract_final_answer(output)
#     if ans is None:
#         return True
#     if len(ans.strip()) == 0: # answer too short
#         return True
#     if len(ans.split()) > 6: # answer too long = model is rambling
#         return True
#     if len(output.split()) > 60: # output too long = drifting away
#         return True
#     return False

def low_confidence(output: str):
    ans = extract_final_answer(output)
    if ans is None:
        return True

    ans = ans.strip().lower()

    if len(ans) == 0:
        return True

    # placeholder / junk outputs
    bad_markers = [
        "<show",
        "<answer>",
        "<option>",
        "step",
        "solution",
        "divide",
        "let x",
        "|question_end|"
    ]
    if any(b in ans for b in bad_markers):
        return True

    # answer too long (rambling)
    if len(ans.split()) > 6:
        return True

    # output contains multiple numbers → unstable
    nums = re.findall(r"-?\d+(\.\d+)?", output)
    if len(nums) > 1:
        return True

    # output too long overall
    if len(output.split()) > 50:
        return True

    return False

results = []
# tasks_small = tasks[1:3]
# print(tasks_small)

for task in tasks:
    q = task["question"]
    answer = normalise(str(task["answer"]))

    prompt = base_prompt(q)
    output = generate(prompt, max_new_tokens=20)
    prediction = normalise(extract_final_answer(output))
    tokens_used = 20

    if prediction is None or low_confidence(output):
        prompt = cot_prompt(q)
        output = generate(prompt, max_new_tokens=120)
        prediction = normalise(extract_final_answer(output))
        tokens_used = 120

    correct = answers_match(prediction, answer)
    print(f"Pred: {prediction} | True: {answer}")
    print("Correct:", correct, "Tokens:", tokens_used)
    print("-" * 60)


    results.append({
        "id": task["id"],
        "output": output,
        "correct": correct,
        "tokens_used": tokens_used
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