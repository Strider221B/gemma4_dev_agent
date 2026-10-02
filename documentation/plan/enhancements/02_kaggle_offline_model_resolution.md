# Enhancement Plan 02 — Kaggle Offline Model Path Resolution

---

## 1. Goal

Eliminate `[Errno -3] Temporary failure in name resolution` during post-training pipeline runs on Kaggle by resolving model identifiers to the local Kaggle mount path `/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2`.

---

## 2. Implementation Steps

### Phase 1: Constants & ModelPathResolver
- [x] Add `KAGGLE_MODEL_PATH` to [`src/utils/constants.py`](file:///home/somesh/git_repos/gemma4_dev_agent/src/utils/constants.py).
- [x] Implement [`ModelPathResolver`](file:///home/somesh/git_repos/gemma4_dev_agent/src/utils/model_path_resolver.py) in `src/utils/model_path_resolver.py`.
- [x] Add `get_resolved_path()` method to [`ModelConfig`](file:///home/somesh/git_repos/gemma4_dev_agent/src/config/model_config.py).

### Phase 2: Pipeline Integration
- [x] Update [`SFTTrainerPipeline._load_model`](file:///home/somesh/git_repos/gemma4_dev_agent/src/training/sft_trainer.py) to resolve model path before loading.
- [x] Update [`RLTrainerPipeline._load_sft_model`](file:///home/somesh/git_repos/gemma4_dev_agent/src/training/rl_trainer.py) to resolve model path before loading.
- [x] Update [`TokenCounter._load_tokenizer`](file:///home/somesh/git_repos/gemma4_dev_agent/src/utils/token_counter.py) to resolve tokenizer path.

### Phase 3: Configuration & Notebooks
- [x] Update [`configs/sft_config.yaml`](file:///home/somesh/git_repos/gemma4_dev_agent/configs/sft_config.yaml) to point to `/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2`.
- [x] Update [`configs/rl_config.yaml`](file:///home/somesh/git_repos/gemma4_dev_agent/configs/rl_config.yaml) to point to `/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2`.
- [x] Update [`notebooks/train_notebook.ipynb`](file:///home/somesh/git_repos/gemma4_dev_agent/notebooks/train_notebook.ipynb) cell 5 to log resolved model path.

### Phase 4: Unit Testing & CI Verification
- [x] Create [`tests/unit/utils/test_model_path_resolver.py`](file:///home/somesh/git_repos/gemma4_dev_agent/tests/unit/utils/test_model_path_resolver.py) covering all resolution scenarios.
- [x] Update [`tests/unit/utils/test_constants.py`](file:///home/somesh/git_repos/gemma4_dev_agent/tests/unit/utils/test_constants.py) for `KAGGLE_MODEL_PATH`.
- [x] Update [`tests/unit/config/test_model_config.py`](file:///home/somesh/git_repos/gemma4_dev_agent/tests/unit/config/test_model_config.py) for `get_resolved_path`.
- [x] Update [`tests/unit/training/test_sft_trainer.py`](file:///home/somesh/git_repos/gemma4_dev_agent/tests/unit/training/test_sft_trainer.py) and [`tests/unit/training/test_rl_trainer.py`](file:///home/somesh/git_repos/gemma4_dev_agent/tests/unit/training/test_rl_trainer.py) to verify path resolution calls.
- [x] Run full CI: `ruff check ./src`, `ruff check ./tests`, `mypy ./src`, `pytest` (460 passed, 97.37% coverage).
