# export_model.py
import torch
from src.model import TransformerForCausalLM
from transformers import AutoTokenizer
from pathlib import Path

device = "cpu"

model_path = Path("model")
tokenizer_path = Path("tokenizer")

model =  TransformerForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

text = 'Zagreb'
input_ids = torch.tensor(tokenizer.encode(text)[:-1], device=device)[None, :]

traced_model = torch.jit.trace(model, (input_ids, ))
torch.jit.save(traced_model, "model.pt")

print("Model exported to model.pt")
