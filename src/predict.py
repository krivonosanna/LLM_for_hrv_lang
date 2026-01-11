import argparse
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model import TransformerForCausalLM
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output CSV file")
    
    args, overrides = parser.parse_known_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model_path = Path("model")
    tokenizer_path = Path("tokenizer")

    print(f"Loading model from {model_path}...")
    model =  TransformerForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    df = pd.read_csv(args.input_path)
    assert "input_model" in df.columns, "Input CSV must contain 'text' column"

    predictions = []
    for text in df["input_model"]:
        input_ids = torch.tensor(tokenizer.encode(text)[:-1], device=device)[None, :]
        model_output = model.generate(
            input_ids, max_new_tokens=200, eos_token_id=tokenizer.eos_token_id, do_sample=True, top_k=10
        )
        output = tokenizer.decode(model_output[0].tolist())
        predictions.append(output)

    # Сохранение
    result_df = pd.DataFrame({"input": df["input_model"], "prediction": predictions})
    result_df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")

if __name__ == "__main__":
    main()
