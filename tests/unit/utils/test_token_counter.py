"""Unit tests for TokenCounter."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.utils.token_counter import TokenCounter


class TestTokenCounter:
    """Test suite for TokenCounter token estimation and budget checks."""

    _SAMPLE_TEXT: str = "def solve_issue():\n    return True\n"
    _SHORT_CHARS: int = 16
    _EXPECTED_SHORT_TOKENS: int = 4
    _TINY_BUDGET: int = 2
    _LARGE_BUDGET: int = 1000

    def test_count_tokens_returns_positive_int(self) -> None:
        """Verify count_tokens returns positive integer for non-empty text."""
        counter = TokenCounter()
        count = counter.count_tokens(self._SAMPLE_TEXT)
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_empty_string_returns_zero(self) -> None:
        """Verify empty string returns zero token count."""
        counter = TokenCounter()
        assert counter.count_tokens("") == 0

    def test_estimate_from_chars_approximation(self) -> None:
        """Verify character estimation rule of 4 chars per token."""
        counter = TokenCounter()
        assert counter.estimate_from_chars(self._SHORT_CHARS) == self._EXPECTED_SHORT_TOKENS
        assert counter.estimate_from_chars(0) == 0
        assert counter.estimate_from_chars(-5) == 0

    def test_fits_budget_within_budget_returns_true(self) -> None:
        """Verify text within budget returns True."""
        counter = TokenCounter()
        assert counter.fits_budget(self._SAMPLE_TEXT, self._LARGE_BUDGET) is True

    def test_fits_budget_exceeds_budget_returns_false(self) -> None:
        """Verify text exceeding budget returns False."""
        counter = TokenCounter()
        assert counter.fits_budget(self._SAMPLE_TEXT, self._TINY_BUDGET) is False

    def test_count_tokens_with_mock_tokenizer(self) -> None:
        """Verify count_tokens uses encode method if tokenizer is loaded."""
        counter = TokenCounter()
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [101, 2054, 102]
        counter._tokenizer = mock_tok

        assert counter.count_tokens("hello") == 3
        mock_tok.encode.assert_called_once_with("hello")

    def test_count_tokens_handles_encode_exception(self) -> None:
        """Verify fallback to char estimation when tokenizer encode raises exception."""
        counter = TokenCounter()
        mock_tok = MagicMock()
        mock_tok.encode.side_effect = RuntimeError("Tokenizer failed")
        counter._tokenizer = mock_tok

        count = counter.count_tokens("fourteen chars")
        assert count > 0

    def test_load_tokenizer_handles_failure(self) -> None:
        """Verify _load_tokenizer gracefully returns None on failed import or load."""
        counter = TokenCounter("nonexistent-model-name")
        assert counter._tokenizer is None
