import os
from dotenv import load_dotenv
load_dotenv()  # This reads your .env file

HF_TOKEN    = os.getenv("HF_TOKEN")
WANDB_KEY   = os.getenv("WANDB_API_KEY")
MODEL_ID    = "mistralai/Mistral-7B-v0.1"
DATASET_ID  = "databricks/databricks-dolly-15k"
LORA_R      = 16
LORA_ALPHA  = 32
LR          = 2e-4
EPOCHS      = 3
BATCH_SIZE  = 4
OUTPUT_DIR  = "./checkpoints"
ADAPTER_DIR = "./adapters/dolly-r16"
