"""Evaluate a trained CodeT5 + LoRA model on the test split."""

import argparse

from codeuctivity.data import build_tokenizer, load_data, preprocess_dataset
from codeuctivity.model.loader import load_trained_model
from codeuctivity.training import build_training_args, run_training
from codeuctivity.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CodeT5 + LoRA")
    parser.add_argument("--config", default="configs/full.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_trained_model(cfg["model"]["output_dir"], cfg["model"]["base_model"])

    dataset = load_data(cfg["data"]["subset"])
    tokenized = preprocess_dataset(
        dataset, tokenizer, cfg["data"]["max_input_len"], cfg["data"]["max_output_len"]
    )
    test_ds = tokenized["test"]

    training_args = build_training_args(output_dir=cfg["model"]["output_dir"])
    from transformers import DataCollatorForSeq2Seq, Trainer

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
