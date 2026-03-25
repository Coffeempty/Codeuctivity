"""Tokenization and preprocessing for CodeT5 code summarization."""

from transformers import AutoTokenizer, PreTrainedTokenizer
from datasets import DatasetDict


def build_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Load tokenizer from HuggingFace hub.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)


def preprocess_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_input_len: int = 128,
    max_output_len: int = 64,
) -> DatasetDict:
    """Tokenize code/docstring pairs with summarize prompt format.

    Args:
        dataset: Raw dataset with 'code' and 'docstring' columns.
        tokenizer: Tokenizer to use for encoding.
        max_input_len: Maximum token length for input code.
        max_output_len: Maximum token length for target docstring.

    Returns:
        Tokenized DatasetDict.
    """

    def _preprocess(batch: dict) -> dict:
        inputs = [f"summarize: {code}" for code in batch["code"]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_len,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            batch["docstring"],
            max_length=max_output_len,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(_preprocess, batched=True)
