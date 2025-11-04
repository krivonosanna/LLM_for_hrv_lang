# tests/test_data.py
import pytest
from datasets import Dataset
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import Tokenizer, decoders
import torch
from src.data import train_tokenizer, chunk_by_document, TextDataset, create_dataloader
from src.model import TransformerForCausalLM
from typing import List

from transformers import AutoTokenizer
from datasets import Dataset

from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class TransformerConfig:
    n_layer: int
    n_head: int
    n_kv_head: int
    hidden_dim: int
    intermediate_dim: int
    dropout: float
    vocab_size: int
    max_seq_len: int


@pytest.fixture
def mock_tokenizer():
    return AutoTokenizer.from_pretrained("tokenizer")


def test_preprocess_valid_text(mock_tokenizer):
    texts = {'texts': ["Zdravo svijete!", "Hrvatska je lijepa."]}
    config = {"model": {"max_seq_len": 100}}
    dataset = chunk_by_document(texts, mock_tokenizer, config)
    assert "input_ids" in dataset
    assert len(dataset["input_ids"]) == 2
    assert isinstance(dataset["input_ids"][0], List)


def test_datasets(mock_tokenizer):
    texts = {'texts': ["Zdravo svijete!", "Hrvatska je lijepa."]}
    config = {"model": {"max_seq_len": 100}}
    dataset = chunk_by_document(texts, mock_tokenizer, config)
    print(dataset)
    train_dataset = TextDataset(dataset, mock_tokenizer)
    assert len(train_dataset) == 2
    assert isinstance(train_dataset[0], List)


def test_dataloader(mock_tokenizer):
    texts = {'texts': ["Zdravo svijete!", "Hrvatska je lijepa."]}
    config = {"model": {"max_seq_len": 100}}
    dataset = chunk_by_document(texts, mock_tokenizer, config)
    train_dataset = TextDataset(dataset, mock_tokenizer)
    train_dataloader = create_dataloader(train_dataset, mock_tokenizer.eos_token_id, max_seq_len=5, batch_size=2, is_train=True)
    ds = next(iter(train_dataloader))
    assert len(ds) == 2
    assert ds[0].shape == (2, 5)
    assert isinstance(ds[0], torch.Tensor)
    assert isinstance(ds[1], torch.Tensor)

def test_model_outputs(mock_tokenizer):
    texts = {'texts': ["Zdravo svijete!", "Hrvatska je lijepa."]}
    config = {"model": {"max_seq_len": 100}}
    dataset = chunk_by_document(texts, mock_tokenizer, config)
    train_dataset = TextDataset(dataset, mock_tokenizer)
    train_dataloader = create_dataloader(train_dataset, mock_tokenizer.eos_token_id, max_seq_len=5, batch_size=2, is_train=True)
    ds = next(iter(train_dataloader))
    
    model_configs = TransformerConfig(n_layer=1,
                                  n_head=2,
                                  n_kv_head=1,
                                  hidden_dim=8,
                                  intermediate_dim=8,
                                  dropout=0,
                                  vocab_size=mock_tokenizer.vocab_size,
                                  max_seq_len=5,
                                  )

    model = TransformerForCausalLM(model_configs)
    input_ids, attention_mask = ds
    out = model(input_ids, attention_mask)
    assert out.shape == (2, 5, mock_tokenizer.vocab_size)
    assert isinstance(out, torch.Tensor)
