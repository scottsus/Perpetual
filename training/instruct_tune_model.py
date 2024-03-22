import os
import torch
from transformers import MambaForCausalLM, AutoTokenizer
from datasets import load_dataset
from ..utils.names import (
    BASE_HF_MODEL, INSTRUCT_DATASET_NAME, INSTRUCT_MODEL_WEIGHTS_FILE
)
from ..utils.prompt import alpaca_prompt as prompt
from ..utils.train import begin_training

def begin_instruction_tuning():
    if os.path.exists(INSTRUCT_MODEL_WEIGHTS_FILE):
        print(f"Model already trained, weights found in `{INSTRUCT_MODEL_WEIGHTS_FILE}`")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.current_device()}")

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }


    dataset = load_dataset(INSTRUCT_DATASET_NAME, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(f"Loaded dataset: {dataset}")

    model_checkpoint = BASE_HF_MODEL
    model = MambaForCausalLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model.to(device)

    begin_training(model, tokenizer, dataset, INSTRUCT_MODEL_WEIGHTS_FILE)
    print(f"✅ Instruction-tuning success, saved weights in `{INSTRUCT_MODEL_WEIGHTS_FILE}`")

if __name__ == "__main__":
    begin_instruction_tuning()