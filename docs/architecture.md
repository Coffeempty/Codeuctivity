# Architecture Notes

## Overview

Codeuctivity fine-tunes [Salesforce/codet5-base](https://huggingface.co/Salesforce/codet5-base) with LoRA adapters for the task of generating natural language docstrings from Python source code.

## Model

- **Base model**: CodeT5-base (encoder-decoder, ~224M parameters)
- **Adapter**: LoRA (r=16, alpha=32, dropout=0.1) applied to attention layers
- **Trainable parameters**: 1,769,472 (0.79% of total 224,651,520)

## Data

- **Dataset**: `code_x_glue_ct_code_to_text` (Python subset, HuggingFace Datasets)
- **Train**: 40,000 samples (subset of 251,820 available)
- **Eval**: 1,000 samples (subset of 13,914 available)
- **Input format**: `"summarize: {code}"` (T5-style task prefix)
- **Max input tokens**: 128, max output tokens: 64

## Training

- **Optimizer**: AdamW (default HuggingFace Trainer)
- **Learning rate**: 5e-4
- **Batch size**: 4 per device
- **Steps**: 10,000 (1 epoch over 40k samples)
- **Mixed precision**: fp16 when CUDA available
- **Logging**: W&B

## Module Layout

```
src/codeuctivity/
    data/           dataset loading and tokenization
    model/          LoRA config and model loading
    training/       HuggingFace Trainer wrapper
    inference/      Summarizer class and convenience function
    utils/          YAML config loader
```
