import torch
from trl import SFTTrainer
from transformers import TrainingArguments

def begin_training(
    model, tokenizer, dataset, model_weights_path,

    # Hyperparameters
    max_seq_len = 2048,
    dataset_num_proc = 2,
    dataset_text_field = "text",
    packing = False,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate = 2e-4,
    warmup_steps = 5,
    logging_steps = 50,
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    optimizer = "adamw_8bit",
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    output_dir = "./results",
):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_len=max_seq_len,
        dataset_text_field=dataset_text_field,
        dataset_num_proc=dataset_num_proc,
        packing=packing,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            optim=optimizer,
            fp16=fp16,
            bf16=bf16,
            output_dir=output_dir,
        )
    )

    trainer.train()
    torch.save(model.state_dict(), model_weights_path)