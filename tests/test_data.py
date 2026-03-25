"""Unit tests for data preprocessing."""


def test_prompt_format() -> None:
    """preprocess_dataset prefixes code inputs with 'summarize: '."""
    received_inputs: list[str] = []

    class FakeTok:
        def __call__(self, texts, **kwargs):
            if isinstance(texts, list):
                received_inputs.extend(texts)
            else:
                received_inputs.append(texts)
            n = len(texts) if isinstance(texts, list) else 1
            return {"input_ids": [[0] * 128] * n, "attention_mask": [[1] * 128] * n}

    from codeuctivity.data.tokenization import preprocess_dataset

    class FakeDatasetDict:
        def map(self, fn, batched=True):
            return fn({"code": ["def f(): pass", "x = 1"], "docstring": ["a fn", "assign"]})

    preprocess_dataset(FakeDatasetDict(), FakeTok())
    assert all(t.startswith("summarize: ") for t in received_inputs[:2])


def test_labels_added() -> None:
    """preprocess_dataset result contains a 'labels' key."""

    class FakeTok:
        def __call__(self, texts, **kwargs):
            n = len(texts) if isinstance(texts, list) else 1
            return {"input_ids": [[0] * 64] * n, "attention_mask": [[1] * 64] * n}

    from codeuctivity.data.tokenization import preprocess_dataset

    class FakeDatasetDict:
        def map(self, fn, batched=True):
            return fn({"code": ["def f(): pass"], "docstring": ["a fn"]})

    result = preprocess_dataset(FakeDatasetDict(), FakeTok())
    assert "labels" in result
