import torch
from transformers import MambaForCausalLM, AutoTokenizer
from .utils.names import (
    BASE_HF_MODEL,
    FINETUNE_MODEL_WEIGHTS_FILE,
)
from .utils.generate import stream

def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.current_device()}")

    model_checkpoint = BASE_HF_MODEL
    model = MambaForCausalLM.from_pretrained(model_checkpoint, state_dict=None)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model.load_state_dict(torch.load(FINETUNE_MODEL_WEIGHTS_FILE))
    model.to(device)

    try:
        print("Welcome to scottsus/mamba's amazing language model 👋")
        while True:
            query = input("> ")
            stream(model, tokenizer, device, query, max_new_tokens=100)
    except KeyboardInterrupt:
        print("👋 Goodbye!")

if __name__ == "__main__":
    run_inference()