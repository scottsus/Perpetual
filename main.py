"""
This training was adapted from the original Python notebook:
https://colab.research.google.com/drive/1D4bs256BFq6PwbiB7TMCWl0Ag5tDeBoi#scrollTo=87tyCRAjYHq9
"""

def train_for_knowledge_injection():
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.current_device(), device)

    from transformers import MambaForCausalLM, AutoTokenizer

    model_checkpoint = "state-spaces/mamba-1.4b-hf"
    model = MambaForCausalLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model.to(device)

    from transformers import TextStreamer

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    def stream(model, tokenizer, query, max_new_tokens = 20):
        inputs = tokenizer(
            [alpaca_prompt.format(query, "", "")],
            return_tensors="pt"
        ).to(device)
        streamer = TextStreamer(tokenizer)
        _ = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            streamer = streamer,
        )

    stream(model, tokenizer, "Who is the current president of the United States?")

    from datasets import load_dataset

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)
    # dataset = dataset.select(range(2_000))

    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Use a fresh pair of weights
    model = MambaForCausalLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    seed = 6699

    max_seq_len = 2048
    dataset_num_proc = 2
    dataset_text_field = "text"
    packing = False

    per_device_train_batch_size = 2
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    warmup_steps = 5
    max_steps = 60
    logging_steps = 50
    weight_decay = 0.01
    lr_scheduler_type = "linear"
    optimizer = "adamw_8bit"

    fp16 = not torch.cuda.is_bf16_supported()
    bf16 = torch.cuda.is_bf16_supported()
    output_dir = "./results"

    def begin_training(dataset, model_weights_path):
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            max_seq_length = max_seq_len,
            dataset_text_field = dataset_text_field,
            dataset_num_proc = dataset_num_proc,
            packing = packing,
            args = TrainingArguments(
                per_device_train_batch_size = per_device_train_batch_size,
                gradient_accumulation_steps = gradient_accumulation_steps,
                learning_rate = learning_rate,
                warmup_steps = warmup_steps,
                # max_steps = max_steps,
                logging_steps = logging_steps,
                weight_decay = weight_decay,
                lr_scheduler_type = lr_scheduler_type,
                optim = optimizer,
                fp16 = fp16,
                bf16 = bf16,
                output_dir = output_dir,
            )
        )
        trainer.train()
        torch.save(model.state_dict(), model_weights_path)

    model_weights_path = "scottsus-mamba-1.4b-hf.pt"
    begin_training(dataset, model_weights_path)

    """
    Reload the model from trained checkpoint
    """

    model = MambaForCausalLM.from_pretrained(model_checkpoint, state_dict=None)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)

    stream(model, tokenizer, "Which is better for Knowledge Injection? Fine-tuning or retrieval?", max_new_tokens = 128)

    """
    We'll be teaching our model about this paper: [Fine-Tuning or Retrieval?
    Comparing Knowledge Injection in LLMs by Ovadia et al](https://arxiv.org/pdf/2312.05934.pdf).
    """

    open_curriculum_dataset = load_dataset("mark-arts/opencurriculumv0", split = "train")

    open_curriculum_model_weights_path = "scottsus-mamba-1.4b-open-curriculum-hf.pt"

    begin_training(open_curriculum_dataset, open_curriculum_model_weights_path)

    """
    Reload the model from the latest checkpoint
    This model should be both fine-tuned for Q&A and also have knowledge about our paper.
    """

    model = MambaForCausalLM.from_pretrained(model_checkpoint, state_dict=None)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model.load_state_dict(torch.load(open_curriculum_model_weights_path))
    model.to(device)

    stream(model, tokenizer, "Which is better for Knowledge Injection? Fine-tuning or retrieval?", max_new_tokens = 128)

if __name__ == "__main__":
    train_for_knowledge_injection()
