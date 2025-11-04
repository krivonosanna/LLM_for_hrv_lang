from datasets import load_dataset, Dataset
import regex as re
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from functools import partial
import torch
from torch.utils.data import DataLoader


def clean_text(ds):
    ds['text'] = ds['text'].lower()
    ds['text'] = re.sub(r"\s+", " ", ds['text']).strip()
    return ds


def load_and_preprocess_data(config):
    data_config = config["data"]

    # Поддержка текстового файла или Hugging Face datasets
    if data_config["use_load_from_txt"]:
        hrv_dataset = load_dataset("texts", data_files=data_config["file_args"]["path"])
        hrv_dataset = hrv_dataset.map(clean_text)
    else:
        hrv_dataset = load_dataset(**data_config["load_args"]) #load_dataset("HuggingFaceFW/fineweb-2", name="hrv_Latn", split="train", streaming=True) 
        hrv_dataset = hrv_dataset.map(clean_text)

        if data_config["load_args"]["streaming"]:
            NUM_SAMPLES = data_config["num_samples"]
            dataset = {'texts': []}

            for i, ds in enumerate(hrv_dataset):
                dataset['texts'].append(ds['text'])
                if i == NUM_SAMPLES:
                    break

            hrv_dataset = Dataset.from_dict(dataset)

    hrv_dataset = hrv_dataset.train_test_split(test_size=data_config["test_size"])
    return hrv_dataset

    
def train_tokenizer(dataset, config):
    texts_for_tokenizer = dataset["texts"][:config["tokenizer"]["num_samples_for_tokenizer"]]
    VOCAB_SIZE = config["model"]["vocab_size"]

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[EOS]", "[UNK]", "[PAD]", "[CLS]",])
    tokenizer.train_from_iterator(texts_for_tokenizer, trainer=trainer, length=len(texts_for_tokenizer))

    tokenizer.post_processor = TemplateProcessing(single="$A [EOS]", special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[EOS]", tokenizer.token_to_id("[EOS]"))])
    

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        eos_token="[EOS]",
    )
    
    return tokenizer


def chunk_by_document(examples, tokenizer, config):
    """
    Разбивает КАЖДЫЙ документ отдельно (сохраняя границы).
    """
    input_ids = []
    block_size = config["model"]["max_seq_len"]

    for text in examples["texts"]:
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        tokens.append(tokenizer.eos_token_id)

        for i in range(0, len(tokens), block_size):
            chunk = tokens[i : i + block_size]
            if len(chunk) < block_size:
                chunk += [tokenizer.pad_token_id] * (block_size - len(chunk))

            input_ids.append(chunk)

    return {"input_ids": input_ids}


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, tokenizer):
        self.tokenizer = tokenizer
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset["input_ids"])

    def __getitem__(self, idx):
        tokenized_sequence = self.tokenized_dataset["input_ids"][idx]
        return tokenized_sequence


def data_collator(
    tokenized_sequences: list[list[int]], pad_token_id: int, max_seq_len: int = None
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(tokenized_sequences)
    max_batch_seq_len = min(max_seq_len, max((len(it) for it in tokenized_sequences)))

    input_ids = torch.full((batch_size, max_batch_seq_len), pad_token_id)
    attention_mask = torch.zeros((batch_size, max_batch_seq_len))

    for i, tok_seq in enumerate(tokenized_sequences):
        cur_len = min(len(tok_seq), max_batch_seq_len)
        input_ids[i, :cur_len] = torch.tensor(tok_seq[:cur_len])
        attention_mask[i, :cur_len] = 1

    return input_ids, attention_mask


def create_dataloader(dataset, pad_token_id, max_seq_len, batch_size, is_train):
    collate_fn = partial(data_collator, pad_token_id=pad_token_id, max_seq_len=max_seq_len)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, drop_last=is_train, collate_fn=collate_fn, pin_memory=True
    )
