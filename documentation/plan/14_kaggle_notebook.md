# Epic 13 — Kaggle Notebook

> **Runs on:** Server (Kaggle 4×L4 GPUs)
> **Depends on:** Epic 12 (Submission Templates)
> **Estimated effort:** ~2 hours
> **Goal:** Create the Kaggle training notebook that imports the `src/` package, runs the full pipeline (data → SFT → RL → CV → package → submit), and produces `submission.zip`.

---

## Pre-Requisites

- All previous epics are complete
- `src/` package is ready for upload to Kaggle
- Activate environment: `source ~/python_envs/p313_llm/bin/activate`

---

## Design Reference

- `high_level/07_deployment_strategy.md` § 4 — Kaggle Notebook Structure
- `high_level/02_architecture_diagram.md` § 3.5 — Data flow

---

## Task 13.1: Create Training Notebook

**File:** `notebooks/train_notebook.ipynb`

This is a Jupyter notebook. Create it as a Python script that will be converted to `.ipynb` format later, OR create the `.ipynb` directly.

**Cell structure:**

### Cell 1: Install our code package
```python
# Install our code from the uploaded Kaggle dataset
!pip install -e /kaggle/input/gemma4-dev-agent-code/src --quiet
```

### Cell 2: Import modules
```python
from src.config.config_manager import ConfigManager
from src.data.dataset_builder import DatasetBuilder
from src.data.ingestor import DataIngestor
from src.data.trajectory_synthesiser import TrajectorySynthesiser
from src.data.trajectory_augmentor import TrajectoryAugmentor
from src.data.chat_formatter import ChatFormatter
from src.data.trajectory_validator import TrajectoryValidator
from src.data.patch_parser import PatchParser
from src.evaluation.cv_splitter import CVSplitter
from src.training.sft_trainer import SFTTrainerPipeline
from src.training.rl_trainer import RLTrainerPipeline
from src.training.checkpoint_manager import CheckpointManager
from src.training.curriculum_scheduler import CurriculumScheduler
from src.training.reward_model import RewardModel
from src.evaluation.cv_evaluator import CVEvaluator
from src.evaluation.metric_tracker import MetricTracker
from src.evaluation.execution_budget import ExecutionBudget
from src.evaluation.trajectory_executor import TrajectoryExecutor
from src.evaluation.verification_runner import VerificationRunner
from src.evaluation.mock_sandbox import MockSandbox
from src.deployment.submission_packager import SubmissionPackager
from src.deployment.constraint_validator import ConstraintValidator
from src.utils.telemetry_logger import TelemetryLogger
from src.utils.token_counter import TokenCounter
```

### Cell 3: Load configuration
```python
config = ConfigManager.load("/kaggle/input/gemma4-dev-agent-code/configs/sft_config.yaml")
telemetry = TelemetryLogger(log_dir="/kaggle/working/logs", run_version="v0.1.0")
```

### Cell 4: Build dataset
```python
# Wire up dependencies
token_counter = TokenCounter()
patch_parser = PatchParser()
ingestor = DataIngestor(config.get_data_paths())
synthesiser = TrajectorySynthesiser(
    ingestor=ingestor,
    patch_parser=patch_parser,
    token_counter=token_counter,
)
augmentor = TrajectoryAugmentor()
formatter = ChatFormatter(token_counter=token_counter)
validator = TrajectoryValidator(token_counter=token_counter)
splitter = CVSplitter()

builder = DatasetBuilder(
    ingestor=ingestor,
    synthesiser=synthesiser,
    augmentor=augmentor,
    formatter=formatter,
    validator=validator,
    splitter=splitter,
)

dataset = builder.build()
print(f"Dataset built: {len(dataset)} examples")
print(f"Columns: {dataset.column_names}")
```

### Cell 5: SFT Training
```python
checkpoint_mgr = CheckpointManager()
curriculum = CurriculumScheduler()

sft_pipeline = SFTTrainerPipeline(
    checkpoint_mgr=checkpoint_mgr,
    telemetry=telemetry,
    curriculum=curriculum,
)

sft_config = config.get_sft_config()
sft_adapter_path = sft_pipeline.run(sft_config, dataset)
print(f"SFT adapter saved to: {sft_adapter_path}")
```

### Cell 6: RL Training (optional — enable when SFT baseline is stable)
```python
# Uncomment when ready for RL training:
# rl_config = config.get_rl_config()
# reward_model = RewardModel(telemetry=telemetry)
# rl_pipeline = RLTrainerPipeline(
#     reward_model=reward_model,
#     checkpoint_mgr=checkpoint_mgr,
#     telemetry=telemetry,
# )
# tasks = ingestor.load_tasks()
# rl_adapter_path = rl_pipeline.run(rl_config, tasks)
# print(f"RL adapter saved to: {rl_adapter_path}")
```

### Cell 7: Quick CV Evaluation (optional)
```python
# Uncomment to run quick CV:
# sandbox = MockSandbox()
# budget = ExecutionBudget()
# executor = TrajectoryExecutor(budget=budget, telemetry=telemetry)
# verifier = VerificationRunner(sandbox=sandbox, telemetry=telemetry)
# tracker = MetricTracker()
#
# evaluator = CVEvaluator(
#     splitter=splitter,
#     executor=executor,
#     verifier=verifier,
#     tracker=tracker,
#     telemetry=telemetry,
# )
# tasks = ingestor.load_tasks()
# report = evaluator.evaluate_all_folds(sft_adapter_path, tasks, version="v0.1.0")
# print(f"CV Resolution Rate: {report.aggregate_resolution_rate:.1%}")
```

### Cell 8: Package Submission
```python
from src.config.deploy_config import DeployConfig

deploy_config = DeployConfig(
    templates_dir="/kaggle/input/gemma4-dev-agent-code/submission_templates",
    output_dir="/kaggle/working/submission",
    zip_path="/kaggle/working/submission.zip",
    adapters=[{
        "name": "coder_lora",
        "checkpoint_path": str(sft_adapter_path),
    }],
)

validator = ConstraintValidator()
packager = SubmissionPackager(validator=validator)
zip_path = packager.package(deploy_config)
print(f"Submission packaged: {zip_path}")

import os
size_mb = os.path.getsize(zip_path) / 1e6
print(f"Submission size: {size_mb:.1f} MB")
```

---

## Task 13.2: Create Kaggle Staging Symlinks/Copies

Ensure the `kaggle_staging/` directory has everything needed for upload:

```bash
cd /home/somesh/git_repos/gemma4_dev_agent

# Copy src/ into kaggle_staging/
cp -r src/ kaggle_staging/src/

# Copy configs
cp -r configs/ kaggle_staging/configs/

# Copy notebooks
cp -r notebooks/ kaggle_staging/notebooks/

# submission_templates already there from Epic 12
```

**Note:** This should be automated by the `VersionManager.push()` method (or done manually before push).

---

## Task 13.3: Update `kaggle_staging/dataset-metadata.json`

Ensure the metadata file has correct content:

```json
{
    "title": "gemma4-dev-agent-code",
    "id": "someshchatterjee/gemma4-dev-agent-code",
    "licenses": [{"name": "CC0-1.0"}]
}
```

---

## Task 13.4: Create Mock Notebook for Local Testing

**File:** `notebooks/mock_notebook.py`

A simplified Python script that runs the pipeline in mock mode (no GPU):

```python
"""Local mock pipeline test — runs without GPU."""

from src.data.mock_data_factory import MockDataFactory
from src.data.dataset_builder import DatasetBuilder
# ... (abbreviated pipeline using build_mock())
```

This script verifies the full pipeline wiring works locally before uploading to Kaggle.

---

## Task 13.5: Run Local Mock Test

```bash
source ~/python_envs/p313_llm/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

python notebooks/mock_notebook.py
```

---

## Completion Criteria

- [ ] `train_notebook.ipynb` exists with all 8 cells
- [ ] Mock notebook runs locally without errors
- [ ] `kaggle_staging/` contains `src/`, `configs/`, `notebooks/`, `submission_templates/`
- [ ] `dataset-metadata.json` has correct Kaggle dataset ID
- [ ] Pipeline wiring is correct (dependencies injected properly)

---

## Files Created in This Epic

```
notebooks/train_notebook.ipynb (or .py that converts to .ipynb)
notebooks/mock_notebook.py
kaggle_staging/src/                    (copy)
kaggle_staging/configs/                (copy)
kaggle_staging/notebooks/              (copy)
```
