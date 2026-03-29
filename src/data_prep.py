from datasets import load_dataset
from transformers import AutoTokenizer
from src.config import MODEL_ID, DATASET_ID, HF_TOKEN

def format_prompt(ex):
    return {
        "text": f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['response']}"
    }

def get_datasets():
    # Download the dataset from Hugging Face
    ds = load_dataset(DATASET_ID, split="train")

    # Format every row into our prompt template
    ds = ds.map(format_prompt)

    # Split: 90% for training, 10% for testing
    ds = ds.train_test_split(test_size=0.1, seed=42)

    # Load the tokenizer (converts words → numbers the model understands)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    train_ds = ds["train"].map(tokenize, batched=True)
    val_ds   = ds["test"].map(tokenize, batched=True)

    return train_ds, val_ds, tokenizer
