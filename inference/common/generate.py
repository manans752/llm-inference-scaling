import torch

def generate(model, tokenizer, prompt: str, max_new_tokens=100, do_sample=False, temperature=0.5, top_p=0.95, repetition_penalty=1.1):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
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