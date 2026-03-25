"""LoRA configuration for CodeT5."""

from peft import LoraConfig, TaskType


def build_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> LoraConfig:
    """Build LoRA adapter config for seq2seq language modeling.

    Args:
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability for LoRA layers.

    Returns:
        Configured LoraConfig.
    """
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
