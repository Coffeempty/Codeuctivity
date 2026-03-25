"""Unit tests for inference module."""

from unittest.mock import MagicMock

import torch

from codeuctivity.inference.summarizer import Summarizer


def _make_mock_model(output_ids: list[int] | None = None) -> MagicMock:
    """Build a minimal mock model for inference tests."""
    model = MagicMock()
    model.device = torch.device("cpu")
    ids = torch.tensor([[0, 42, 7, 2]]) if output_ids is None else torch.tensor([output_ids])
    model.generate.return_value = ids
    return model


def _make_mock_tokenizer(decoded: str = "returns n factorial") -> MagicMock:
    tok = MagicMock()
    # BatchEncoding supports .to(); simulate it with a MagicMock that returns itself
    encoding = MagicMock()
    encoding.to.return_value = encoding
    tok.return_value = encoding
    tok.decode.return_value = decoded
    return tok


def test_summarizer_returns_string() -> None:
    """Summarizer.summarize returns a non-empty string."""
    model = _make_mock_model()
    tok = _make_mock_tokenizer("returns n factorial")
    s = Summarizer(model, tok, max_output_len=64)
    result = s.summarize("def factorial(n): return 1 if n==0 else n*factorial(n-1)")
    assert isinstance(result, str)
    assert len(result) > 0


def test_summarizer_calls_generate() -> None:
    """Summarizer calls model.generate exactly once."""
    model = _make_mock_model()
    tok = _make_mock_tokenizer()
    s = Summarizer(model, tok, max_output_len=32)
    s.summarize("def f(): pass")
    model.generate.assert_called_once()


def test_summarizer_prompt_format() -> None:
    """Summarizer passes 'summarize: <code>' to tokenizer."""
    model = _make_mock_model()
    tok = _make_mock_tokenizer()
    s = Summarizer(model, tok)
    s.summarize("def f(): pass")
    call_args = tok.call_args
    assert call_args[0][0] == "summarize: def f(): pass"
