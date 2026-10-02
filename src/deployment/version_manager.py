"""Version manager for dataset releases and Kaggle CLI interaction."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from typing import ClassVar


class VersionManager:
    """Manages semantic versioning, changelog updates, and Kaggle pushes."""

    _VERSION_FILE: ClassVar[str] = "VERSION"
    _CHANGELOG_FILE: ClassVar[str] = "CHANGELOG.md"
    _VALID_BUMP_TYPES: ClassVar[frozenset[str]] = frozenset({"major", "minor", "patch"})
    _DEFAULT_VERSION: ClassVar[str] = "0.1.0"
    _DEPLOYMENT_LOG_FILE: ClassVar[str] = "deployment.log"
    _KAGGLE_CMD: ClassVar[str] = "kaggle"
    _PREFIX_V: ClassVar[str] = "v"
    _VERSION_DELIMITER: ClassVar[str] = "."

    def __init__(self, staging_dir: str) -> None:
        """Initialize VersionManager with a staging directory.

        Args:
            staging_dir: Root directory where VERSION and CHANGELOG reside.
        """
        self._staging_dir = staging_dir

    def bump(self, bump_type: str, message: str) -> str:
        """Increment version and record changelog message.

        Args:
            bump_type: One of 'major', 'minor', 'patch'.
            message: Description of the changes for the new release.

        Returns:
            The newly bumped semantic version string.
        """
        if bump_type not in self._VALID_BUMP_TYPES:
            raise ValueError(
                f"Invalid bump_type '{bump_type}'. Must be one of {sorted(self._VALID_BUMP_TYPES)}"
            )
        current = self._read_version()
        major, minor, patch = self._parse_version(current)
        if bump_type == "major":
            new_version = f"{major + 1}.0.0"
        elif bump_type == "minor":
            new_version = f"{major}.{minor + 1}.0"
        else:
            new_version = f"{major}.{minor}.{patch + 1}"
        self._write_version(new_version)
        self._update_changelog(new_version, message)
        return new_version

    def get_current_version(self) -> str:
        """Read and return current version from VERSION file.

        Returns:
            Current version string.
        """
        return self._read_version()

    def push(self, staging_dir: str, version: str, message: str) -> None:
        """Push dataset staging directory to Kaggle via Kaggle CLI.

        Args:
            staging_dir: Directory containing staged files to push.
            version: Version string tag.
            message: Commit/version message.
        """
        cmd = [
            self._KAGGLE_CMD,
            "datasets",
            "version",
            "-p",
            staging_dir,
            "-m",
            f"{self._PREFIX_V}{version}: {message}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Kaggle push failed: {result.stderr.strip()}")
        self._log_deployment(version, message)

    def _read_version(self) -> str:
        version_path = os.path.join(self._staging_dir, self._VERSION_FILE)
        if not os.path.isfile(version_path):
            return self._DEFAULT_VERSION
        with open(version_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content if content else self._DEFAULT_VERSION

    def _write_version(self, version: str) -> None:
        os.makedirs(self._staging_dir, exist_ok=True)
        version_path = os.path.join(self._staging_dir, self._VERSION_FILE)
        with open(version_path, "w", encoding="utf-8") as f:
            f.write(f"{version}\n")

    def _update_changelog(self, version: str, message: str) -> None:
        cl_path = os.path.join(self._staging_dir, self._CHANGELOG_FILE)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entry = f"\n## [{self._PREFIX_V}{version}] - {timestamp}\n- {message}\n"
        with open(cl_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def _parse_version(self, version: str) -> tuple[int, int, int]:
        clean = version.lstrip(self._PREFIX_V).strip()
        parts = clean.split(self._VERSION_DELIMITER)
        if len(parts) != 3:
            raise ValueError(f"Invalid semantic version format: '{version}'")
        return int(parts[0]), int(parts[1]), int(parts[2])

    def _log_deployment(self, version: str, message: str) -> None:
        log_path = os.path.join(self._staging_dir, self._DEPLOYMENT_LOG_FILE)
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = f"[{timestamp}] {self._PREFIX_V}{version}: {message}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(entry)
