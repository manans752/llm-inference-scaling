import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from inference.common.common import MODEL_NAME, HF_TOKEN

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        token=HF_TOKEN
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer

def load_model(tokenizer):
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        token=HF_TOKEN
    )
    config.pad_token_id = config.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=HF_TOKEN
    )

    model.eval()
    return model