import torch
from transformers import MambaForCausalLM, AutoTokenizer
from datasets import load_dataset
from scripts.utils.names import (
    BASE_HF_MODEL, INSTRUCT_DATASET_NAME, INSTRUCT_MODEL_HF_NAME
)
from scripts.utils.prompt import alpaca_prompt as prompt
from scripts.utils.train import begin_training

def begin_instruction_tuning():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {torch.cuda.current_device()}")

    model_checkpoint = BASE_HF_MODEL
    model = MambaForCausalLM.from_pretrained(model_checkpoint, state_dict=None)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model.to(device)

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

    dataset = load_dataset(INSTRUCT_DATASET_NAME, split="train").train_test_split(test_size=0.15)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    print(f"Loaded {INSTRUCT_DATASET_NAME} dataset: {dataset}")

    begin_training(model, tokenizer, train_dataset, eval_dataset)
    
    # Requires notebook_login()
    model.push_to_hub(INSTRUCT_MODEL_HF_NAME)
    print(f"✅ Instruction-tuning success, find the model on `{INSTRUCT_MODEL_HF_NAME}`")

if __name__ == "__main__":
    begin_instruction_tuning()
