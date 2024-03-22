import os
import torch
from transformers import MambaForCausalLM, AutoTokenizer
from datasets import load_dataset
from ..utils.names import (
    INSTRUCT_MODEL_WEIGHTS_FILE,
    FINETUNE_MODEL_WEIGHTS_FILE,
    BASE_HF_MODEL,
    FINETUNE_DATASET_NAME,
)
from ..utils.train import begin_training

def begin_fine_tuning():
    if not os.path.exists(INSTRUCT_MODEL_WEIGHTS_FILE):
        print(f"Model not yet instruction-tuned. Please run `python training/instruct_tune_model.py`.")
        return
    
    if os.path.exists(FINETUNE_MODEL_WEIGHTS_FILE):
        print(f"Model already fine-tuned, weights found in `{FINETUNE_MODEL_WEIGHTS_FILE}`")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.current_device()}")

    dataset = load_dataset(FINETUNE_DATASET_NAME, split="train")
    print(f"Loaded dataset: {dataset}")

    model_checkpoint = BASE_HF_MODEL
    model = MambaForCausalLM.from_pretrained(model_checkpoint, state_dict=None)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    model.load_state_dict(torch.load(INSTRUCT_MODEL_WEIGHTS_FILE))
    model.to(device)

    begin_training(model, tokenizer, dataset, FINETUNE_MODEL_WEIGHTS_FILE)
    print(f"✅ Fine-tuning success, saved weights in `{FINETUNE_MODEL_WEIGHTS_FILE}`")

if __name__ == "__main__":
    begin_fine_tuning()