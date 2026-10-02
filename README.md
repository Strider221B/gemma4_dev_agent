# SweGemma-Agent

> **Autonomous Software Engineering Agent Post-Training & Submission Pipeline**  
> Solution for the [Kaggle Google - Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent) competition.

---

## 1. Overview

`SweGemma-Agent` is a complete, modular, and leak-free post-training and deployment pipeline that adapts Google's Gemma 4 (`gemma-4-31b-it-qat-w4a16-ct`) into an autonomous software engineering developer agent capable of resolving GitHub issues in complex repositories.

### Key Capabilities
- **Synthetic Trajectory Engine**: Reconstructs realistic multi-turn agent exploration and editing trajectories from issue descriptions, git commits, unified diff patches, and call graphs.
- **Data Augmentation & Token Budgeting**: Generates error-recovery and negative-example trajectories while strictly enforcing maximum context window budgets.
- **Supervised Fine-Tuning (SFT)**: Employs QLoRA parameter-efficient fine-tuning via Unsloth/TRL with difficulty-aware curriculum scheduling.
- **Reinforcement Learning Alignment (RL)**: Supports GRPO/DPO preference optimization using automated test execution, patch validity, and tool efficiency rewards.
- **Group K-Fold Cross-Validation (CV)**: Enforces zero-data-leakage evaluation grouped strictly by repository to mirror competition evaluation.
- **Adversarial & Peer Inspection**: Analyzes peer solutions and parses runtime notebooks to identify anti-patterns, data leakage, and prompt injection vulnerabilities.
- **Strict Pre-Flight Packaging**: Compiles system prompts and validates all competition constraints (Safetensors format, LoRA rank $\le 128$, $\le 8$ adapters, $\le 4\text{ GB}$ total size, single model declarations) before creating `submission.zip`.

---

## 2. High-Level Design & Architecture Documentation

Detailed design specifications and architectural blueprints are maintained in [`documentation/design/high_level/`](documentation/design/high_level/):

| Document | Focus | Description |
|---|---|---|
| [**01_system_overview.md**](documentation/design/high_level/01_system_overview.md) | System Overview | Scope, constraints, hardware assumptions, and high-level requirements. |
| [**02_architecture_diagram.md**](documentation/design/high_level/02_architecture_diagram.md) | Architecture Diagram | Component layout, inter-module dependencies, and pipeline lifecycle. |
| [**03_data_flow_pipeline.md**](documentation/design/high_level/03_data_flow_pipeline.md) | Data Pipeline | Task ingestion, trajectory synthesis, and chat formatting with special tokens. |
| [**04_training_strategy.md**](documentation/design/high_level/04_training_strategy.md) | Training Strategy | QLoRA configurations, curriculum scheduling, DPO, and GRPO reward models. |
| [**05_evaluation_framework.md**](documentation/design/high_level/05_evaluation_framework.md) | Evaluation Framework | Repository-grouped CV splitting, sandbox execution, and overfitting metrics. |
| [**06_agent_architecture.md**](documentation/design/high_level/06_agent_architecture.md) | Agent Architecture | Agent tree hierarchy, tool schemas, prompt compiler, and nudges. |
| [**07_deployment_strategy.md**](documentation/design/high_level/07_deployment_strategy.md) | Deployment Strategy | Kaggle notebook lifecycle, submission packaging, and constraint checks. |
| [**08_risk_mitigation.md**](documentation/design/high_level/08_risk_mitigation.md) | Risk Mitigation | Failure modes, context overflow prevention, and timeout mitigations. |

For coding conventions and implementation rules, see [**coding_standards.md**](documentation/prompts/coding_standards.md).

---

## 3. Repository Structure

```
gemma4_dev_agent/
├── configs/                       # Pipeline & training configuration YAML files
│   ├── eval_config.yaml           # Evaluation & CV fold parameters
│   ├── peer_registry.yaml         # Peer solution inspection registry
│   ├── rl_config.yaml             # RL / GRPO / DPO training parameters
│   └── sft_config.yaml            # SFT, model, LoRA, and curriculum parameters
├── documentation/                 # Architectural specifications & execution plans
│   ├── design/high_level/         # System design specifications
│   ├── plan/                      # Step-by-step epic implementation plans
│   └── prompts/                   # Adspire coding standards & guidelines
├── kaggle_staging/                # Clean bundle for Kaggle Dataset upload
│   ├── dataset-metadata.json      # Kaggle dataset definition
│   ├── submission_templates/      # agent.yaml, prompts, sub-agents, and skills
│   └── VERSION                    # Semantic release version tracking
├── notebooks/                     # Interactive & automation notebooks
│   ├── mock_notebook.py           # Local end-to-end dry-run pipeline test (no GPU)
│   └── train_notebook.ipynb       # Production Kaggle training notebook (4×L4 GPUs)
├── src/                           # Production Python package (swegemma-agent)
│   ├── config/                    # Pydantic schemas & YAML configuration manager
│   ├── data/                      # Ingestion, synthesis, augmentation, & validation
│   ├── deployment/                # Submission packager, prompt compiler, & validator
│   ├── evaluation/                # CV evaluator, splitter, sandbox, & peer analysis
│   ├── training/                  # SFT, RL, checkpoints, & curriculum scheduler
│   └── utils/                     # Telemetry logger, token counter, git utilities
├── tests/                         # Comprehensive pytest test suite (>97% coverage)
│   └── unit/                      # Unit tests structured per package module
├── pyproject.toml                 # Package metadata, dependencies, & tool configs
└── setup.py                       # Editable installation bootstrap
```

---

## 4. Setup & Installation

### Prerequisites
- Linux OS (Ubuntu 22.04+ or WSL2 recommended)
- Python 3.13+
- `uv` (recommended) or standard `pip`

### Step 1: Activate Environment
```bash
# Using the preconfigured virtual environment:
source ~/python_envs/p313_llm/bin/activate

# Or create a new virtual environment using uv:
uv venv ~/python_envs/p313_llm --python 3.13
source ~/python_envs/p313_llm/bin/activate
```

### Step 2: Install in Editable Mode
```bash
cd /home/somesh/git_repos/gemma4_dev_agent
uv pip install -e . --no-deps
```

---

## 5. Local Development, Testing & Verification

Every component in this repository adheres to strict Object-Oriented Programming (OOP) and passes three quality gates before deployment:

### 1. Run the Local Mock Pipeline (Dry-Run)
Verifies the complete end-to-end pipeline wiring (data synthesis, curriculum filtering, safetensors checkpointing, CV fold splitting, and submission zip generation) in $< 5\text{ seconds}$ without requiring GPUs:
```bash
python notebooks/mock_notebook.py
```

### 2. Run Quality Checks (CI Pipeline)
Run the linting, type-checking, and test suite:

```bash
# 1. Lint code and notebooks (0 errors allowed)
ruff check src/ tests/ notebooks/

# 2. Strict static type check (0 errors allowed)
mypy src/

# 3. Run full test suite with coverage report (>= 90% required)
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 6. Kaggle Staging & Dataset Synchronization

To run on Kaggle, the `src/` package, `configs/`, `notebooks/`, and `submission_templates/` must be staged into `kaggle_staging/` and uploaded as a private Kaggle dataset.

### Step 1: Sync Workspace to Staging Area
Sync repository files while automatically excluding temporary `__pycache__` artifacts:

```bash
cd /home/somesh/git_repos/gemma4_dev_agent

# Sync src package
rsync -av --delete --exclude '__pycache__' src/ kaggle_staging/src/

# Sync configuration files
rsync -av --delete --exclude '__pycache__' configs/ kaggle_staging/configs/

# Sync notebooks
rsync -av --delete --exclude '__pycache__' notebooks/ kaggle_staging/notebooks/
```

### Step 2: Validate `dataset-metadata.json`
Verify `kaggle_staging/dataset-metadata.json` points to your Kaggle username and dataset ID:
```json
{
  "title": "gemma4-dev-agent-code",
  "id": "someshchatterjee/gemma4-dev-agent-code",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
```

### Step 3: Push Dataset via Kaggle CLI
You can use `VersionManager` or the Kaggle CLI directly:
```bash
# Create dataset for the first time:
kaggle datasets create -p kaggle_staging/ --dir-mode zip

# Or bump version and push new release:
kaggle datasets version -p kaggle_staging/ --dir-mode zip -m "v0.1.0: update pipeline and training notebook"
```

---

## 7. Running the Training Notebook on Kaggle

### 1. Kaggle Environment Setup
1. Create a new notebook on Kaggle targeting the **Google - Gemma 4 Developer Agent** competition.
2. Select Accelerator: **4 × NVIDIA L4 GPUs** (or 2 × T4 / A100).
3. Attach Datasets via **Add Input**:
   - `gemma-4-developer-agent` (Official competition data)
   - `someshchatterjee/gemma4-dev-agent-code` (Your uploaded code package dataset)
4. Import [`notebooks/train_notebook.ipynb`](notebooks/train_notebook.ipynb) or copy its 8 cells into the Kaggle notebook.

### 2. Notebook Execution Stages

| Cell | Stage | Actions Performed |
|---|---|---|
| **Cell 1** | **Register Package Path** | Adds `/kaggle/input/gemma4-dev-agent-code` to `sys.path` to import `src`. |
| **Cell 2** | **Import Modules** | Imports [`ConfigManager`](src/config/config_manager.py), [`DatasetBuilder`](src/data/dataset_builder.py), [`SFTTrainerPipeline`](src/training/sft_trainer.py), [`SubmissionPackager`](src/deployment/submission_packager.py), etc. |
| **Cell 3** | **Load Config** | Reads `/kaggle/input/gemma4-dev-agent-code/configs/sft_config.yaml` and initializes `/kaggle/working/logs`. |
| **Cell 4** | **Build Dataset** | Ingests `tasks.jsonl`, graphs, and embeddings; synthesizes trajectories and assigns CV folds. |
| **Cell 5** | **SFT Training** | Loads 4-bit Gemma 4 base model, applies QLoRA adapters, runs curriculum training, and saves checkpoint to `/kaggle/working/checkpoints/sft_lora/`. |
| **Cell 6** | **RL Training** | *(Optional)* Runs preference optimization and GRPO reward-driven reinforcement learning. |
| **Cell 7** | **CV Evaluation** | *(Optional)* Evaluates out-of-fold resolution rates using mock sandboxes and task verifiers. |
| **Cell 8** | **Package Submission** | Assembles templates from `submission_templates/` and trained adapters into `/kaggle/working/submission/`, validates pre-flight constraints, and produces `/kaggle/working/submission.zip`. |

### 3. Generated Artifacts
Upon completion, the notebook produces:
- `/kaggle/working/checkpoints/sft_lora/` (`adapter_model.safetensors`, `adapter_config.json`)
- `/kaggle/working/submission.zip` (Ready for submission to the competition evaluation harness)
- `/kaggle/working/logs/run_v0.1.0.log` (Full training telemetry and metrics)

---

## 8. Submission Rules & Constraint Checks

The [`ConstraintValidator`](src/deployment/constraint_validator.py) automatically enforces all competition constraints during packaging:

1. **Root Configuration**: `agent.yaml` must exist in root and declare schema version, models, and tool definitions.
2. **Total Size Limit**: Uncompressed staging directory must not exceed $4\text{ GB}$ ($4{,}294{,}967{,}296\text{ bytes}$).
3. **No Legacy Binaries**: Checkpoint directories must only contain `.safetensors` files (`.bin`, `.pt`, `.pth` are rejected).
4. **Adapter Count**: No more than 8 adapter checkpoints may be referenced.
5. **LoRA Rank**: LoRA rank $r$ must not exceed 128 across all adapters.
6. **Single Base Model**: Exactly one distinct base model must be declared across `agent.yaml` and sub-agent configs.
7. **No Symlinks**: Hard copies only — absolute and relative symlinks are strictly rejected.
8. **Token Limits**: Generation configs must define explicit `max_output_tokens` and `thinking_budget`.

---

## 9. License

This repository is licensed under the [Apache License 2.0](LICENSE).
The dataset metadata is released under [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/).
