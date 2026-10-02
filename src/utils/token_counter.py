"""Token estimation and budget accounting utilities."""

from __future__ import annotations

import math
from typing import cast


class TokenCounter:
    """Estimator for token counts and context budget verification."""

    _CHARS_PER_TOKEN: float = 4.0
    _MIN_TOKEN_COUNT: int = 0

    def __init__(self, tokenizer_name: str | None = None) -> None:
        """Initialize TokenCounter optionally with a named tokenizer model."""
        self._tokenizer_name: str | None = tokenizer_name
        self._tokenizer: object | None = (
            self._load_tokenizer(tokenizer_name) if tokenizer_name else None
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens using loaded tokenizer if available or character heuristic."""
        if not text:
            return self._MIN_TOKEN_COUNT
        if self._tokenizer is not None and hasattr(self._tokenizer, "encode"):
            try:
                tokens = getattr(self._tokenizer, "encode")(text)
                return len(tokens)
            except Exception:
                return self.estimate_from_chars(len(text))
        return self.estimate_from_chars(len(text))

    def estimate_from_chars(self, char_count: int) -> int:
        """Estimate token count from character length using 4 chars per token rule."""
        if char_count <= 0:
            return self._MIN_TOKEN_COUNT
        return max(1, math.ceil(char_count / self._CHARS_PER_TOKEN))

    def fits_budget(self, text: str, budget: int) -> bool:
        """Check whether given text token count is within the allocated budget."""
        return self.count_tokens(text) <= budget

    def _load_tokenizer(self, name: str) -> object | None:
        """Attempt to load a tokenizer instance by identifier."""
        try:
            from transformers import AutoTokenizer

            from src.utils.model_path_resolver import ModelPathResolver

            resolved_path = ModelPathResolver.resolve(name)
            return cast(object, AutoTokenizer.from_pretrained(resolved_path))
        except Exception:
            return None
