# train.py
import argparse
import yaml
import logging
from src.data import load_and_preprocess_data, train_tokenizer, chunk_by_document, TextDataset, create_dataloader
from src.model import TransformerForCausalLM
from src.train import Trainer
from src.utils import set_seed, setup_logger
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


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    # args = parser.parse_args()

    args, overrides = parser.parse_known_args()


    # with open(args.config, "r", encoding="utf-8") as f:
    #     config = yaml.safe_load(f)

    config = OmegaConf.load(args.config)
    cli_conf = OmegaConf.from_dotlist(overrides)
    config = OmegaConf.merge(config, cli_conf)
    # print(OmegaConf.to_yaml(config))

    set_seed(config["seed"])

    logger = setup_logger(name=__name__, log_file=config["logging"]["log_dir"], level=logging.INFO)

    logger.info("Loading datatsets...")
    dataset = load_and_preprocess_data(config)
    
    logger.info("Train tokenizer...")
    tokenizer = train_tokenizer(dataset["train"], config)

    logger.info("Save tokenizer...")
    tokenizer.save_pretrained(config["tokenizer"]["path_save"])

    logger.info("Preprocessing data...")
    tokenized_dataset_train = dataset['train'].map(lambda examples: chunk_by_document(examples, tokenizer, config),
                                     batched=True,
                                     remove_columns=dataset['train'].column_names
                                )
    
    tokenized_dataset_val = dataset['test'].map(lambda examples: chunk_by_document(examples, tokenizer, config),
                                     batched=True,
                                     remove_columns=dataset['test'].column_names
                                )
    

    train_dataset = TextDataset(tokenized_dataset_train, tokenizer)
    train_dataloader = create_dataloader(train_dataset, tokenizer.eos_token_id, max_seq_len=config["model"]["max_seq_len"], batch_size=config["trainer"]["batch_size"], is_train=True)

    val_dataset = TextDataset(tokenized_dataset_val, tokenizer)
    val_dataloader = create_dataloader(val_dataset, tokenizer.eos_token_id, max_seq_len=config["model"]["max_seq_len"], batch_size=config["trainer"]["batch_size"], is_train=False)
    
    model_configs = TransformerConfig(n_layer=config["model"]["n_layer"],
                                  n_head=config["model"]["n_head"],
                                  n_kv_head=config["model"]["n_kv_head"],
                                  hidden_dim=config["model"]["hidden_dim"],
                                  intermediate_dim=config["model"]["intermediate_dim"],
                                  dropout=config["model"]["dropout"],
                                  vocab_size=config["model"]["vocab_size"],
                                  max_seq_len=config["model"]["max_seq_len"],
                                  )

    model = TransformerForCausalLM(model_configs)
    trainer = Trainer(config, logger)

    logger.info("Starting training pipeline...")
    trainer.run(model, train_dataloader, val_dataloader)

    logger.info("Training completed successfully!")
    
    logger.info("Save model")
    model.save_pretrained(config["model"]["path_save"])
    

if __name__ == "__main__":
    main()
