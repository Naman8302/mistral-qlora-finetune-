import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.config import MODEL_ID, ADAPTER_DIR, HF_TOKEN

app = FastAPI(title="Mistral QLoRA API — github.com/Naman8302")

# This class defines what a request to the API looks like
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200

# Load model once when the server starts (not on every request)
model, tokenizer = None, None

@app.on_event("startup")
def load_model_on_start():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto",
        token=HF_TOKEN
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    print("Model loaded and ready!")

@app.get("/")
def root():
    return {"message": "Mistral QLoRA API is running. POST to /generate"}

@app.post("/generate")
def generate(req: GenerateRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=req.max_tokens)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"prompt": req.prompt, "response": response_text}
