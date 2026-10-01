"""Unit tests for deployment layer constants."""

from __future__ import annotations

import src.deployment.constants as deploy_constants


class TestDeploymentConstants:
    """Test suite for deployment layer constants."""

    _EXPECTED_SIZE_LIMIT: int = 3_221_225_472

    def test_deployment_layer_constants(self) -> None:
        """Verify deployment size limits and file extension sets."""
        assert deploy_constants.MAX_TOTAL_SIZE_BYTES == self._EXPECTED_SIZE_LIMIT
        assert "agent.yaml" in deploy_constants.REQUIRED_FILES
        assert ".yaml" in deploy_constants.ALLOWED_EXTENSIONS
        assert ".pkl" in deploy_constants.FORBIDDEN_EXTENSIONS
        assert deploy_constants.VERSION_FILE == "VERSION"
