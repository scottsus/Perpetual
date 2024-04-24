from transformers import TextStreamer
from .prompt import alpaca_prompt as prompt

def stream(model, tokenizer, device, query, max_new_tokens=20):
    inputs = tokenizer(
        [prompt.format(query, "", "")],
        return_tensors="pt"
    ).to(device)

    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    _ = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
    )
