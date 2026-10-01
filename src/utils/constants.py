"""Shared utility constants used across multiple modules."""

from __future__ import annotations

# Model identifier
MODEL_NAME: str = "google/gemma-4-31b-it-qat-w4a16-ct"

# Submission constraints
MAX_SUBMISSION_SIZE_BYTES: int = 3_221_225_472  # 3 GiB
MAX_ADAPTERS: int = 8
MAX_LORA_RANK: int = 128
MAX_CONTEXT_WINDOW: int = 32_768

# Budget defaults
DEFAULT_MAX_TOOL_CALLS: int = 100
DEFAULT_MAX_TIME_MINUTES: float = 60.0
DEFAULT_COMMAND_TIMEOUT_SECONDS: int = 300
DEFAULT_MAX_STDOUT_CHARS: int = 5_000
DEFAULT_MAX_FILE_LINES: int = 150
DEFAULT_MAX_FILE_CHARS: int = 10_000

# Token budgets
MAX_TRAJECTORY_TOKENS: int = 28_672
THINKING_BUDGET_TOKENS: int = 4_096
COMPACTION_THRESHOLD_TOKENS: int = 14_336

# Adapter constraints
ADAPTER_SIZE_LIMIT_BYTES: int = 1_500_000_000  # 1.5 GiB safety
ALLOWED_ADAPTER_EXTENSIONS: frozenset[str] = frozenset({".safetensors", ".json"})
FORBIDDEN_EXTENSIONS: frozenset[str] = frozenset(
    {".bin", ".pt", ".pth", ".pkl", ".pickle"}
)
