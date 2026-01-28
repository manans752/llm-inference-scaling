import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

model.eval()

def format_prompt(question: str) -> str:
    return f"<s>[INST] {question} [/INST]"



def tokenize_prompt(prompt: str):
    return tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    )