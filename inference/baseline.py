import json
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pathlib import Path

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

MODEL_NAME = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"Pad token ID: {tokenizer.pad_token_id}")

HERE = Path(__file__).parent
PATH_TO_TASKS = HERE.parent / "data" / "tasks.json"

with PATH_TO_TASKS.open("r", encoding="utf-8") as f:
    tasks = json.load(f)

print(f"Confirm loaded {len(tasks)} tasks.")

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

# model.generation_config.pad_token_id = tokenizer.pad_token_id
# model.generation_config.eos_token_id = tokenizer.eos_token_id

# print(f"Model generation config pad_token_id: {model.generation_config.pad_token_id}")
model.eval()

def format_prompt(question: str) -> str:
    # return question.strip()
    return f"Question: {question.strip()}\nAnswer:"


# print("tokenizer is:", tokenizer)
# print("type:", type(tokenizer))

#returns a dictionary
def tokenize_prompt(prompt: str):
    return tokenizer(
        prompt,
        return_tensors="pt"
    )

def single_pass_generate(prompt: str, max_new_tokens: int):
    inputs = tokenize_prompt(prompt)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    # Debug: print input
    print(f"Input: {tokenizer.decode(inputs['input_ids'][0])}")
    print(f"Input length: {input_len}")

    with torch.no_grad(): #Disabling gradient calculation, reduce memory consumption for computations that would otherwise have `requires_grad=True
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=50256,  # Explicitly set to 50256
            eos_token_id=50256,
            repetition_penalty=1.2,  # Prevent getting stuck on same token
            use_cache=True,
            temperature=None,  # b/c do_sample=False
            top_p=None  # b/c do_sample=False
        )

        # outputs = model.generate(
        #     input_ids=inputs["input_ids"],
        #     attention_mask=inputs.get("attention_mask"),
        #     max_new_tokens=max_new_tokens,
        #     do_sample=False,
        #     pad_token_id=tokenizer.pad_token_id,
        #     eos_token_id=tokenizer.eos_token_id
        #     # **inputs,
        #     # max_new_tokens=max_new_tokens, #fixed compute budget
        #     # do_sample=False,    # no randomness
        #     # pad_token_id=tokenizer.pad_token_id,
        #     # eos_token_id=tokenizer.eos_token_id
        # )
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


MAX_NEW_TOKENS = 50
results = []

for i, task in enumerate(tasks):
    prompt = format_prompt(task["question"])
    output = single_pass_generate(prompt, MAX_NEW_TOKENS)

    correct = str(task["answer"]) in output
    print(f"Task {i}: {task['question']}: {task['answer']}: output: {output}")

    results.append({
        "id": task["id"],
        "output": output,
        "correct": correct,
        "tokens": MAX_NEW_TOKENS
    })

    print(f"[{i+1}/{len(tasks)}] correct={correct}")

with open("results/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Baseline run complete.")
