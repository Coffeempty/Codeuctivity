"""HuggingFace Trainer wrapper with W&B logging."""

from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)


def build_training_args(
    output_dir: str | Path,
    batch_size: int = 4,
    learning_rate: float = 5e-4,
    num_train_epochs: int = 1,
    logging_steps: int = 50,
    max_steps: int = -1,
) -> TrainingArguments:
    """Build TrainingArguments from config values.

    Args:
        output_dir: Directory to write checkpoints and logs.
        batch_size: Per-device train and eval batch size.
        learning_rate: Optimizer learning rate.
        num_train_epochs: Number of full training epochs.
        logging_steps: Log metrics every N steps.
        max_steps: If > 0, overrides num_train_epochs.

    Returns:
        Configured TrainingArguments.
    """
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        logging_dir=str(Path(output_dir) / "logs"),
        logging_steps=logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        fp16=torch.cuda.is_available(),
    )


def run_training(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    training_args: TrainingArguments,
) -> Trainer:
    """Run training and return the Trainer instance.

    Args:
        model: PEFT-wrapped model.
        tokenizer: Tokenizer for data collation.
        train_dataset: Training split.
        eval_dataset: Evaluation split.
        training_args: HuggingFace TrainingArguments.

    Returns:
        Trainer after training completes.
    """
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    return trainer
