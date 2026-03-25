"""Dataset loading utilities for code_x_glue_ct_code_to_text."""

from datasets import DatasetDict, load_dataset


def load_data(subset: str = "python") -> DatasetDict:
    """Load the CodeXGlue code-to-text dataset.

    Args:
        subset: Language subset to load. Defaults to "python".

    Returns:
        DatasetDict with train/validation/test splits.
    """
    return load_dataset("code_x_glue_ct_code_to_text", subset)
