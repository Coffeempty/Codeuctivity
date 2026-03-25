"""Train CodeT5 + LoRA on code-to-text summarization."""

import argparse
from pathlib import Path

from codeuctivity.data import build_tokenizer, load_data, preprocess_dataset
from codeuctivity.model import build_lora_config, load_base_model, load_peft_model
from codeuctivity.training import build_training_args, run_training
from codeuctivity.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CodeT5 + LoRA")
    parser.add_argument("--config", default="configs/full.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = build_tokenizer(cfg["model"]["base_model"])
    dataset = load_data(cfg["data"]["subset"])
    tokenized = preprocess_dataset(
        dataset, tokenizer, cfg["data"]["max_input_len"], cfg["data"]["max_output_len"]
    )

    base_model = load_base_model(cfg["model"]["base_model"])
    lora_cfg = build_lora_config(**cfg["lora"])
    model = load_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = tokenized["train"].select(range(cfg["data"]["train_samples"]))
    eval_ds = tokenized["validation"].select(range(cfg["data"]["eval_samples"]))
    training_args = build_training_args(
        output_dir=cfg["model"]["output_dir"],
        **cfg["training"],
    )
    trainer = run_training(model, tokenizer, train_ds, eval_ds, training_args)
    tokenizer.save_pretrained(cfg["model"]["output_dir"])
    print(f"Saved to {cfg['model']['output_dir']}")


if __name__ == "__main__":
    main()
