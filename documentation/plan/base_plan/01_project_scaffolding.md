# Epic 0 — Project Scaffolding

> **Runs on:** Local machine
> **Depends on:** Nothing
> **Estimated effort:** ~1 hour
> **Goal:** Create the complete directory structure, `pyproject.toml`, and all `__init__.py` files so subsequent epics can immediately start writing classes.

---

## Pre-Requisites

```bash
source ~/python_envs/p312_kaggle/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent
```

---

## Task 0.1: Create Directory Structure

Create every directory shown in `documentation/design/high_level/02_architecture_diagram.md` § 2.

```bash
# Source package
mkdir -p src/config
mkdir -p src/data
mkdir -p src/training
mkdir -p src/evaluation
mkdir -p src/deployment
mkdir -p src/utils

# Tests (mirror src/)
mkdir -p tests/unit/data
mkdir -p tests/unit/training
mkdir -p tests/unit/evaluation
mkdir -p tests/unit/deployment
mkdir -p tests/unit/utils
mkdir -p tests/integration

# Configs
mkdir -p configs

# Logs
mkdir -p logs

# Kaggle staging sub-dirs
mkdir -p kaggle_staging/src
mkdir -p kaggle_staging/configs
mkdir -p kaggle_staging/notebooks
mkdir -p kaggle_staging/submission_templates/prompts
mkdir -p kaggle_staging/submission_templates/sub_agents
mkdir -p kaggle_staging/submission_templates/skills/repo_navigation

# Notebooks
mkdir -p notebooks
```

---

## Task 0.2: Create `__init__.py` Files

Create an empty `__init__.py` in every Python package directory.

**Files to create (all empty):**
```
src/__init__.py
src/config/__init__.py
src/data/__init__.py
src/training/__init__.py
src/evaluation/__init__.py
src/deployment/__init__.py
src/utils/__init__.py
tests/__init__.py
tests/unit/__init__.py
tests/unit/data/__init__.py
tests/unit/training/__init__.py
tests/unit/evaluation/__init__.py
tests/unit/deployment/__init__.py
tests/unit/utils/__init__.py
tests/integration/__init__.py
```

---

## Task 0.3: Create `pyproject.toml`

Create `/home/somesh/git_repos/gemma4_dev_agent/pyproject.toml` with the following content:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "swegemma-agent"
version = "0.1.0"
description = "Post-training pipeline for gemma-4-31b-it-qat-w4a16-ct as an autonomous SWE agent"
requires-python = ">=3.13"
dependencies = [
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "numpy>=1.26",
    "networkx>=3.0",
    "datasets>=2.0",
    "scikit-learn>=1.3",
]

[project.optional-dependencies]
training = [
    "torch>=2.0",
    "transformers>=4.40",
    "peft>=0.12",
    "trl>=0.9",
    "unsloth",
    "safetensors",
]
dev = [
    "ruff>=0.4",
    "mypy>=1.10",
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-mock>=3.0",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "W", "F", "I"]

[tool.mypy]
strict = true
plugins = ["pydantic.mypy"]
mypy_path = "."
packages = ["src"]

[[tool.mypy.overrides]]
module = [
    "unsloth.*",
    "trl.*",
    "peft.*",
    "transformers.*",
    "datasets.*",
    "networkx.*",
    "sklearn.*",
    "safetensors.*",
    "torch.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src --cov-report=term-missing --cov-report=xml -v"

[tool.coverage.run]
source = ["src"]
branch = true

[tool.coverage.report]
fail_under = 90
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.",
    "if TYPE_CHECKING:",
    "\\.\\.\\.",
]
```

---

## Task 0.4: Create `setup.py`

Create `/home/somesh/git_repos/gemma4_dev_agent/setup.py`:

```python
"""Editable install support for swegemma-agent."""
from setuptools import setup

if __name__ == "__main__":
    setup()
```

---

## Task 0.5: Create `tests/conftest.py`

Create `/home/somesh/git_repos/gemma4_dev_agent/tests/conftest.py`:

```python
"""Shared test fixtures for the SweGemma-Agent test suite."""
from __future__ import annotations

import pytest


# Shared fixtures will be added by subsequent epics
```

---

## Task 0.6: Install Package in Editable Mode

```bash
source ~/python_envs/p312_kaggle/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent
pip install -e ".[dev]"
```

---

## Task 0.7: Install Linting/Type-Checking Tools

```bash
source ~/python_envs/p312_kaggle/bin/activate
pip install ruff mypy pytest pytest-cov pytest-mock pydantic
```

---

## Task 0.8: Verify Setup

Run these commands and confirm zero errors:

```bash
source ~/python_envs/p312_kaggle/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

# Should pass with no files to check yet
ruff check src/ tests/

# Should pass (empty package)
mypy src/

# Should collect 0 tests (none written yet)
pytest tests/ -v
```

---

## Completion Criteria

- [ ] All directories from Task 0.1 exist
- [ ] All `__init__.py` files from Task 0.2 exist
- [ ] `pyproject.toml` exists with correct content
- [ ] `setup.py` exists
- [ ] `tests/conftest.py` exists
- [ ] `pip install -e ".[dev]"` succeeds
- [ ] `ruff check src/` returns 0 errors
- [ ] `mypy src/` returns 0 errors
- [ ] `pytest tests/` runs successfully (0 tests collected is fine)

---

## Files Created in This Epic

```
pyproject.toml                         (new)
setup.py                               (new)
src/__init__.py                        (new)
src/config/__init__.py                 (new)
src/data/__init__.py                   (new)
src/training/__init__.py               (new)
src/evaluation/__init__.py             (new)
src/deployment/__init__.py             (new)
src/utils/__init__.py                  (new)
tests/__init__.py                      (new)
tests/conftest.py                      (new)
tests/unit/__init__.py                 (new)
tests/unit/data/__init__.py            (new)
tests/unit/training/__init__.py        (new)
tests/unit/evaluation/__init__.py      (new)
tests/unit/deployment/__init__.py      (new)
tests/unit/utils/__init__.py           (new)
tests/integration/__init__.py          (new)
```
