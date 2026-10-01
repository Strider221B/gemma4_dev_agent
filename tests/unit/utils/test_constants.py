"""Unit tests for shared utility constants."""

from __future__ import annotations

import src.utils.constants as utils_constants


class TestConstants:
    """Test suite for shared utility constants."""

    _EXPECTED_MODEL: str = "google/gemma-4-31b-it-qat-w4a16-ct"
    _EXPECTED_CONTEXT: int = 32768
    _EXPECTED_RANK: int = 128

    def test_model_constants(self) -> None:
        """Verify model name and dimension constants."""
        assert utils_constants.MODEL_NAME == self._EXPECTED_MODEL
        assert utils_constants.MAX_CONTEXT_WINDOW == self._EXPECTED_CONTEXT
        assert utils_constants.MAX_LORA_RANK == self._EXPECTED_RANK

    def test_budget_and_size_constants(self) -> None:
        """Verify budget and adapter size limit constants."""
        assert utils_constants.DEFAULT_MAX_TOOL_CALLS == 100
        assert utils_constants.DEFAULT_MAX_TIME_MINUTES == 60.0
        assert utils_constants.MAX_TRAJECTORY_TOKENS == 28672
        assert ".safetensors" in utils_constants.ALLOWED_ADAPTER_EXTENSIONS
        assert ".pt" in utils_constants.FORBIDDEN_EXTENSIONS
