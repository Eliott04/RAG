import torch

def build_prompt(tokenizer, user: str, system: str = "You are a helpful assistant.") -> str:
    """Build a chat-format prompt using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate_text(tokenizer, model, device,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
    num_return_sequences: int = 1,
) -> list[str]:
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode and extract only the response
    generated = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        # Extract the part after the last "assistant"
        if "assistant" in text.lower():
            text = text.split("assistant")[-1].strip()
        generated.append(text)

    return generated