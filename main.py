import torch
from transformers import MambaForCausalLM, AutoTokenizer
from src.utils.names import INSTRUCT_MODEL_HF_NAME
from src.utils.generate import stream

def run_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.current_device()}")

    model_checkpoint = INSTRUCT_MODEL_HF_NAME
    model = MambaForCausalLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
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