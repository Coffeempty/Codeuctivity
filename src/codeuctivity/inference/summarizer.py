"""Code summarization inference helpers."""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, PreTrainedTokenizer


class Summarizer:
    """Generates natural language summaries of code snippets.

    Args:
        model: Loaded PEFT model.
        tokenizer: Corresponding tokenizer.
        max_output_len: Maximum tokens to generate.
    """

    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        max_output_len: int = 64,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_output_len = max_output_len

    def summarize(self, code: str) -> str:
        """Generate a docstring summary for the given code.

        Args:
            code: Source code string to summarize.

        Returns:
            Generated natural language summary.
        """
        inputs = self.tokenizer(
            f"summarize: {code}",
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device)
        output = self.model.generate(**inputs, max_length=self.max_output_len)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


_default_summarizer: Summarizer | None = None


def _load_default_summarizer(adapter_path: str | Path = "outputs/codet5-lora") -> Summarizer:
    """Load the default trained summarizer (lazy singleton).

    Args:
        adapter_path: Path to saved adapter weights.

    Returns:
        Initialized Summarizer instance.
    """
    from ..model.loader import load_trained_model

    model, tokenizer = load_trained_model(adapter_path, "Salesforce/codet5-base")
    return Summarizer(model, tokenizer)


def summarize(code: str, adapter_path: str | Path = "outputs/codet5-lora") -> str:
    """Convenience function: summarize code using the default trained model.

    Args:
        code: Source code string to summarize.
        adapter_path: Path to saved adapter weights.

    Returns:
        Generated natural language summary.
    """
    global _default_summarizer
    if _default_summarizer is None:
        _default_summarizer = _load_default_summarizer(adapter_path)
    return _default_summarizer.summarize(code)
