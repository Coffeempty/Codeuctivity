"""Model loading utilities."""

from pathlib import Path

import torch
from peft import PeftModel, get_peft_model
from peft.config import PeftConfigMixin
from transformers import AutoModelForSeq2SeqLM, PreTrainedModel


def load_base_model(model_name: str) -> PreTrainedModel:
    """Load a base seq2seq model from HuggingFace hub.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded model.
    """
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)


def load_peft_model(base_model: PreTrainedModel, lora_config: PeftConfigMixin) -> PeftModel:
    """Wrap a base model with LoRA adapters.

    Args:
        base_model: The base pretrained model.
        lora_config: LoRA configuration.

    Returns:
        PEFT-wrapped model with trainable LoRA adapters.
    """
    return get_peft_model(base_model, lora_config)


def load_trained_model(adapter_path: str | Path, base_model_name: str) -> tuple[PeftModel, object]:
    """Load a trained LoRA adapter from disk.

    Args:
        adapter_path: Path to saved adapter weights.
        base_model_name: HuggingFace identifier for the base model.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoTokenizer

    adapter_path = Path(adapter_path)
    base = load_base_model(base_model_name)
    model = PeftModel.from_pretrained(base, str(adapter_path))
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer
