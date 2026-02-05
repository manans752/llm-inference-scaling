import json
import torch
import pandas as pd
import re
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
            top_p=0.9, #guarantees we only pick from 'reasonable' tokens
            repetition_penalty=1.1 #ensure we don't get stuck in loops
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

# def extract_final_number(text: str) -> str:
#     text = text.strip()
#
#     # Find all integers or decimals
#     numbers = re.findall(r"-?\d+\.?\d*", text)
#
#     if not numbers:
#         return ""
#
#     return numbers[-1]  # last number should be the final answer
#
# def majority_vote(outputs):
#     answers = [extract_final_number(o) for o in outputs]
#     answers = [x for x in answers if x != ""]
#
#     if len(answers) == 0:
#         return "", answers
#
#     vote = Counter(answers)
#     final_answer = vote.most_common(1)[0][0]
#
#     return final_answer, answers

def majority_vote(predictions):
    vote = Counter(predictions)
    return vote.most_common(1)[0][0]

results = []
# tasks_small = tasks[1:3]
# print(tasks_small)
for task in tasks:
    q = task["question"]
    answer = normalise(str(task["answer"]))

    prompt = (
        f"Problem: {task['question']}\n"
        "Solve step by step.\n"
        "End with: Final Answer: <answer>\n"
    )

    for n in N_SAMPLES:
        samples = [sample(prompt) for _ in range(n)]

        final_answer = majority_vote(samples)

        correct = answers_match(final_answer, answer)


        #disagreement across extracted answers
        variance = len(set(samples)) / len(samples)

        results.append({
            "id": task["id"],
            "N": n,
            "final_answer": final_answer,
            "correct": correct,
            "variance": variance
        })

        print(f"Task={task['id']} N={n}")
        # print("Extracted answers:", extracted_answers)
        print("Final vote:", final_answer)
        print("Correct:", correct)
        print("Variance:", variance)

        # print(f"N={n} correct={correct} variance={variance:.2f}")

PATH_TO_RESULTS = HERE.parent / "results" / "self_consistency_results.json"

with PATH_TO_RESULTS.open("w") as f:
    json.dump(results, f, indent=2)

df = pd.DataFrame(results)
df.to_csv(RESULTS_PATH, index=False)
print("Saved to", RESULTS_PATH)

print("\nAccuracy by sample count:")
print(df.groupby("N")["correct"].mean())