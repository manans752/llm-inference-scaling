from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import InferenceClient
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from inference.common.config import ModelRuntimeConfig


@dataclass
class HuggingFaceAPIModel:
    model_name: str
    provider: str
    task_mode: str
    endpoint_url: str | None
    api_base_url: str
    token: str | None
    timeout_seconds: int
    wait_for_model: bool
    client: InferenceClient
    backend: str = "huggingface_api"


def resolve_device(runtime_config: ModelRuntimeConfig) -> str:
    if runtime_config.backend == "huggingface_api":
        return "remote_api"
    if runtime_config.device != "auto":
        return runtime_config.device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(runtime_config: ModelRuntimeConfig, device: str) -> str | torch.dtype:
    if runtime_config.backend == "huggingface_api":
        return "remote_api"
    if runtime_config.dtype == "float32":
        return torch.float32
    if runtime_config.dtype == "float16":
        return torch.float16
    if runtime_config.dtype == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.float16
    return torch.float32


def load_tokenizer(runtime_config: ModelRuntimeConfig) -> Any:
    if runtime_config.backend == "huggingface_api":
        return None

    tokenizer = AutoTokenizer.from_pretrained(
        runtime_config.model_name,
        token=runtime_config.hf_token,
        trust_remote_code=runtime_config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(runtime_config: ModelRuntimeConfig, tokenizer: Any) -> Any:
    if runtime_config.backend == "huggingface_api":
        client_kwargs = {
            "api_key": runtime_config.hf_token,
            "timeout": runtime_config.api_timeout_seconds,
        }
        if runtime_config.endpoint_url:
            client_kwargs["base_url"] = runtime_config.endpoint_url
        else:
            client_kwargs["provider"] = runtime_config.api_provider

        return HuggingFaceAPIModel(
            model_name=runtime_config.api_model_name,
            provider=runtime_config.api_provider,
            task_mode=runtime_config.api_task_mode,
            endpoint_url=runtime_config.endpoint_url,
            api_base_url=runtime_config.api_base_url,
            token=runtime_config.hf_token,
            timeout_seconds=runtime_config.api_timeout_seconds,
            wait_for_model=runtime_config.api_wait_for_model,
            client=InferenceClient(**client_kwargs),
        )

    device = resolve_device(runtime_config)
    dtype = resolve_dtype(runtime_config, device)
    config = AutoConfig.from_pretrained(
        runtime_config.model_name,
        token=runtime_config.hf_token,
        trust_remote_code=runtime_config.trust_remote_code,
    )
    config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        runtime_config.model_name,
        config=config,
        torch_dtype=dtype,
        token=runtime_config.hf_token,
        trust_remote_code=runtime_config.trust_remote_code,
        low_cpu_mem_usage=runtime_config.low_cpu_mem_usage,
    )
    model.to(device)
    model.eval()
    return model
