from __future__ import annotations

import math
import re
import time
from typing import Any

import torch
from huggingface_hub.errors import BadRequestError
from huggingface_hub.inference._generated.types.chat_completion import ChatCompletionOutput
from huggingface_hub.inference._generated.types.text_generation import TextGenerationOutput

from inference.common.config import GenerationConfig
from inference.common.model_setup import HuggingFaceAPIModel


def _estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(re.findall(r"\S+", text)) * 1.3))


def _generate_local(model: Any, tokenizer: Any, prompt: str, generation_config: GenerationConfig) -> dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[1])

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=generation_config.max_new_tokens,
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature if generation_config.do_sample else None,
            top_p=generation_config.top_p if generation_config.do_sample else None,
            repetition_penalty=generation_config.repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    duration = time.perf_counter() - start

    generated_tokens = output_ids[0][prompt_tokens:]
    generated_count = int(generated_tokens.shape[0])
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return {
        "text": decoded,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_count,
        "total_tokens": prompt_tokens + generated_count,
        "latency_seconds": duration,
        "token_count_source": "tokenizer",
    }


def _extract_api_generated_text(payload: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(payload, TextGenerationOutput):
        details = payload.details
        return payload.generated_text, {
            "generated_tokens": getattr(details, "generated_tokens", None),
            "prefill": getattr(details, "prefill", None),
        }

    if isinstance(payload, ChatCompletionOutput):
        first_choice = payload.choices[0]
        content = first_choice.message.content or ""
        usage = payload.usage
        return str(content), {
            "generated_tokens": getattr(usage, "completion_tokens", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    if isinstance(payload, list):
        first = payload[0] if payload else {}
        if not isinstance(first, dict):
            return str(first), {}
        return str(first.get("generated_text", "")), first.get("details", {}) or {}

    if "generated_text" in payload:
        return str(payload.get("generated_text", "")), payload.get("details", {}) or {}

    if isinstance(payload.get("choices"), list) and payload["choices"]:
        first_choice = payload["choices"][0]
        if isinstance(first_choice, dict):
            if "text" in first_choice:
                return str(first_choice["text"]), {}
            if isinstance(first_choice.get("message"), dict):
                return str(first_choice["message"].get("content", "")), {}

    raise RuntimeError(f"Unexpected Hugging Face API response shape: {payload}")


def _generate_huggingface_api(model: HuggingFaceAPIModel, prompt: str, generation_config: GenerationConfig) -> dict[str, Any]:
    start = time.perf_counter()
    mode_used = model.task_mode
    model_arg = {} if model.endpoint_url else {"model": model.model_name}

    def run_text_generation() -> Any:
        return model.client.text_generation(
            prompt,
            **model_arg,
            details=True,
            do_sample=generation_config.do_sample,
            max_new_tokens=generation_config.max_new_tokens,
            repetition_penalty=generation_config.repetition_penalty,
            return_full_text=False,
            temperature=generation_config.temperature if generation_config.do_sample else None,
            top_p=generation_config.top_p if generation_config.do_sample else None,
        )

    def run_chat_completion() -> Any:
        return model.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            **model_arg,
            max_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature if generation_config.do_sample else None,
            top_p=generation_config.top_p if generation_config.do_sample else None,
        )

    if model.task_mode == "chat_completion":
        payload = run_chat_completion()
        mode_used = "chat_completion"
    else:
        try:
            payload = run_text_generation()
            mode_used = "text_generation"
        except (ValueError, BadRequestError) as exc:
            error_text = str(exc).lower()
            if model.task_mode == "text_generation":
                raise
            if "supported task: conversational" not in error_text and "chat-completion" not in error_text and "conversational" not in error_text:
                raise
            payload = run_chat_completion()
            mode_used = "chat_completion"

    duration = time.perf_counter() - start
    generated_text, details = _extract_api_generated_text(payload)

    prompt_tokens = int(
        details.get("prompt_tokens")
        or (details.get("prefill", []) and len(details["prefill"]))
        or _estimate_token_count(prompt)
    )
    generated_tokens = int(details.get("generated_tokens") or _estimate_token_count(generated_text))
    total_tokens = int(details.get("total_tokens") or (prompt_tokens + generated_tokens))

    return {
        "text": generated_text.strip(),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": duration,
        "token_count_source": "api_details" if details else "estimated",
        "api_task_mode_used": mode_used,
    }


def generate(model: Any, tokenizer: Any, prompt: str, generation_config: GenerationConfig) -> dict[str, Any]:
    if getattr(model, "backend", None) == "huggingface_api":
        return _generate_huggingface_api(model, prompt, generation_config)
    return _generate_local(model, tokenizer, prompt, generation_config)
