# Implementation Plan: Handle Pre-quantized Models

## Objective

Fix the `ValueError` encountered when loading models pre-quantized with `CompressedTensorsConfig` (like `gemma-4-31b-it-qat-w4a16-ct`) in the training pipelines by conditionally applying `BitsAndBytesConfig`.

## Tasks

1. **Update `src/training/sft_trainer.py`**
   - Import `AutoConfig` from `transformers`.
   - Modify `_load_model` to load the model's configuration first.
   - Check if `quantization_config` exists on the model configuration.
   - Build a dictionary of `kwargs` for `AutoModelForCausalLM.from_pretrained()`.
   - Add `BitsAndBytesConfig` to `kwargs` only if the model is not natively quantized and 4-bit loading is requested.

2. **Update `src/training/rl_trainer.py`**
   - Import `AutoConfig` from `transformers`.
   - Modify `_load_sft_model` with the exact same conditional quantization logic as in `sft_trainer.py`.

3. **Update Unit Tests (if necessary)**
   - Verify if `test_sft_trainer.py` and `test_rl_trainer.py` fail due to missing `AutoConfig` mocks.
   - Apply `patch("transformers.AutoConfig.from_pretrained")` if required by the test framework.

4. **Verify Implementation**
   - Run type checks (`mypy`, `ruff`).
   - Run test suite to ensure regressions are not introduced.
