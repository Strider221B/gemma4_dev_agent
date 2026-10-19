"""Integration test for local Gemma 4 E2B QAT W4A16 smoke test execution."""

from __future__ import annotations

import tempfile
from typing import ClassVar

import pytest

from src.training.local_model_smoke_tester import LocalModelSmokeTester
from src.utils.constants import LOCAL_TEST_MODEL_NAME
from src.utils.model_path_resolver import ModelPathResolver


class TestLocalModelSmoke:
    """Integration test suite executing real local Gemma 4 E2B model verification when available."""

    _MODEL_NAME: ClassVar[str] = LOCAL_TEST_MODEL_NAME
    _STATUS_KEY: ClassVar[str] = "status"
    _STATUS_SUCCESS: ClassVar[str] = "success"
    _FORWARD_KEY: ClassVar[str] = "forward_passed"
    _BACKWARD_KEY: ClassVar[str] = "backward_passed"
    _ADAPTER_SAVED_KEY: ClassVar[str] = "adapter_saved"

    @pytest.mark.skipif(
        not ModelPathResolver.has_local_model(LOCAL_TEST_MODEL_NAME),
        reason="Local Gemma 4 E2B weights not present on disk",
    )
    def test_local_model_smoke_end_to_end_when_weights_present(self) -> None:
        """Verify real Gemma 4 E2B model completes forward and backward passes without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tester = LocalModelSmokeTester(model_name=self._MODEL_NAME, work_dir=tmpdir)
            report = tester.run_verification()

            assert report[self._STATUS_KEY] == self._STATUS_SUCCESS
            assert report[self._FORWARD_KEY] is True
            assert report[self._BACKWARD_KEY] is True
            assert report[self._ADAPTER_SAVED_KEY] is True
