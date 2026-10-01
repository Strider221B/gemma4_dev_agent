"""Deployment layer constants."""

from __future__ import annotations

MAX_TOTAL_SIZE_BYTES: int = 3_221_225_472
REQUIRED_FILES: list[str] = ["agent.yaml"]
ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".yaml",
        ".yml",
        ".md",
        ".txt",
        ".py",
        ".json",
        ".safetensors",
    }
)
FORBIDDEN_EXTENSIONS: frozenset[str] = frozenset(
    {".bin", ".pt", ".pth", ".pkl", ".pickle"}
)
VERSION_FILE: str = "VERSION"
