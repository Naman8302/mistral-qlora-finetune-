import json
from evaluate import load
from peft import PeftModel
from src.config import ADAPTER_DIR
from src.model import load_model
from src.data_prep import get_datasets

def run_eval():
    _, val_ds, tokenizer = get_datasets()

    # Load the fine-tuned model
    base_model = load_model()
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    # Load evaluation metrics
    rouge      = load("rouge")
    bertscore  = load("bertscore")

    predictions, references = [], []

    # Generate 200 predictions on validation examples
    for ex in val_ds.select(range(200)):
        inputs = tokenizer(ex["text"][:300], return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=150)
        predictions.append(tokenizer.decode(output[0], skip_special_tokens=True))
        references.append(ex["response"])

    # Calculate scores
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bs_results   = bertscore.compute(predictions=predictions,
                                      references=references, lang="en")
    results = {
        **rouge_scores,
        "bertscore_f1": round(sum(bs_results["f1"]) / len(bs_results["f1"]), 4)
    }

    # Save results to a file
    with open("results/eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete!")
    print(results)

if __name__ == "__main__":
    run_eval()
