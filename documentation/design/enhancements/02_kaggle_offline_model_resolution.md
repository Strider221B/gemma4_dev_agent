# Enhancement Design 02 — Kaggle Offline Model Path Resolution

---

## 1. Problem Statement

When executing the post-training pipeline on Kaggle:

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

The run fails with:

```text
'[Errno -3] Temporary failure in name resolution' thrown while requesting HEAD https://huggingface.co/google/gemma-4-31b-it-qat-w4a16-ct/resolve/main/config.json
Retrying in 1s [Retry 1/5].
RuntimeError: Cannot send a request, as the client has been closed.
```

### Root Cause

1. In Kaggle competition notebooks, internet access is strictly disabled.
2. `config.get_sft_config()` loads `sft_config.yaml` or default `ModelConfig(name="google/gemma-4-31b-it-qat-w4a16-ct")`.
3. When `AutoModelForCausalLM.from_pretrained(config.name)` and `AutoTokenizer.from_pretrained(config.name)` are called with `google/gemma-4-31b-it-qat-w4a16-ct`, HuggingFace Transformers interprets this string as a remote Hub repository ID because it does not exist as a local directory on disk.
4. Transformers initiates an HTTPS network request to `huggingface.co`, which immediately fails due to DNS resolution failure in the offline container.
5. In Kaggle notebooks, the model is mounted locally at:
   `/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2`

---

## 2. Solution: ModelPathResolver & Offline Config Defaults

### 2.1 Component Architecture

We introduce a dedicated resolver class and update configuration files and pipelines:

1. **`src/utils/constants.py`**:
   - Define `KAGGLE_MODEL_PATH = "/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2"`.
2. **`src/utils/model_path_resolver.py` (`ModelPathResolver`)**:
   - Implements robust path resolution priority:
     - Direct existing local path -> returns input path.
     - Environment variable `MODEL_PATH` or `GEMMA_MODEL_PATH` -> returns env path if exists.
     - Competition model identifier (`google/gemma-4-31b-it-qat-w4a16-ct` or `gemma-4-31b-it-qat-w4a16-ct`) -> returns `KAGGLE_MODEL_PATH` if mounted on disk.
     - Fallback -> returns original identifier (for local mocking or online fallback).
3. **`src/training/sft_trainer.py`**:
   - `_load_model` resolves model path before passing to `AutoModelForCausalLM` and `AutoTokenizer`.
4. **`src/training/rl_trainer.py`**:
   - `_load_sft_model` resolves model path before passing to `AutoModelForCausalLM` and `AutoTokenizer`.
5. **`src/utils/token_counter.py`**:
   - `_load_tokenizer` resolves tokenizer path before calling `AutoTokenizer.from_pretrained`.
6. **`src/config/model_config.py`**:
   - Adds `get_resolved_path()` helper method.
7. **`configs/sft_config.yaml` & `configs/rl_config.yaml`**:
   - Point `model.name` directly to `/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2`.

---

## 3. Class Design & SOLID Compliance

### Single Responsibility
`ModelPathResolver` has a single responsibility: resolving model identifiers to existing local directory paths.

### Adspire Coding Standards
- Strictly OOP: class with classmethods and helper methods; no free-standing functions.
- All methods <= 20 lines.
- Public method `resolve` appears before protected helper methods.
- Named constants used without magic strings or numbers.
