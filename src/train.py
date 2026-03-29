import wandb
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from src.config import *
from src.model import load_model
from src.data_prep import get_datasets

def train():
    # Log into Weights & Biases
    wandb.login(key=WANDB_KEY)
    wandb.init(
        project="mistral-qlora-finetune",
        name="dolly-r16-run1",
        config={"learning_rate": LR, "epochs": EPOCHS, "lora_r": LORA_R}
    )

    # Load data and model
    train_ds, val_ds, tokenizer = get_datasets()
    model = load_model()

    # All training settings
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",          # Send all metrics to W&B automatically
    )

    # Start training
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()

    # Save only the small LoRA adapter (~50MB, not the full 14GB model)
    model.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved to {ADAPTER_DIR}")
    wandb.finish()

if __name__ == "__main__":
    train()
