# Enhancement Design 02 — Handle Pre-quantized Models Elegantly

---

## 1. Problem Statement

Following the removal of the Unsloth dependency, the project now relies on the standard HuggingFace `transformers` stack for model loading. However, an issue arises when loading pre-quantized models, such as `google/gemma-4-31b-it-qat-w4a16-ct`, which are quantized using `CompressedTensorsConfig`.

The current implementation unconditionally instantiates and passes a `BitsAndBytesConfig` when `config.load_in_4bit` is set (or implicitly). The `transformers` library attempts to merge this provided quantization configuration with the model's native quantization configuration, resulting in a type mismatch error:

```
ValueError: The model is quantized with CompressedTensorsConfig but you are passing a BitsAndBytesConfig config. Please make sure to pass the same quantization config class to `from_pretrained` with different loading attributes.
```

### Affected Source Files

| File | Usages |
|------|----------------|
| `src/training/sft_trainer.py` | `_load_model()` |
| `src/training/rl_trainer.py` | `_load_sft_model()` |

---

## 2. Solution: Conditional Quantization Configuration

To resolve this, we will inspect the model's configuration prior to loading it. We will only provide the `BitsAndBytesConfig` if the model does not already contain a native quantization configuration.

### 2.1 Why This Works

- **Native Support:** `transformers` will automatically apply the model's native quantization (e.g., `CompressedTensorsConfig`, `AWQConfig`, `GPTQConfig`) if it is defined in the `config.json` of the model repository.
- **Robustness:** By omitting the `quantization_config` argument for already quantized models, we avoid the merge conflict.
- **Backward Compatibility:** Standard, unquantized models will continue to use `BitsAndBytesConfig` to dynamically quantize weights at load time, preserving our intended 4-bit memory efficiency.

---

## 3. Detailed Replacement Design

### 3.1 Model Loading Modifications

We will utilize `transformers.AutoConfig` to load the model's configuration metadata before instantiating the model itself.

**Before:**
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=config.load_in_4bit,
    ...
)
model = AutoModelForCausalLM.from_pretrained(
    resolved_path,
    quantization_config=quantization_config,
    ...
)
```

**After:**
```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 1. Load configuration to check for pre-existing quantization
model_config = AutoConfig.from_pretrained(resolved_path, trust_remote_code=True)
has_native_quantization = hasattr(model_config, "quantization_config") and model_config.quantization_config is not None

kwargs = {
    "device_map": self._DEVICE_MAP,
    "trust_remote_code": True,
    "torch_dtype": self._resolve_dtype(config.dtype),
}

# 2. Conditionally apply BitsAndBytesConfig
if not has_native_quantization and config.load_in_4bit:
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=self._resolve_dtype(config.dtype),
        bnb_4bit_quant_type=self._BNB_QUANT_TYPE,
        bnb_4bit_use_double_quant=True,
    )

# 3. Load model
model = AutoModelForCausalLM.from_pretrained(
    resolved_path,
    **kwargs
)
```

### 3.2 Updating `sft_trainer.py` and `rl_trainer.py`

- **`sft_trainer.py` (`_load_model`)**: Update to include the `AutoConfig` check and construct loading `kwargs` dynamically.
- **`rl_trainer.py` (`_load_sft_model`)**: Apply the same logic for base model loading prior to PEFT merging.

---

## 4. Test Impact

Tests targeting `_load_model` and `_load_sft_model` might need minor adjustments to mock `AutoConfig.from_pretrained`. If the mocked configuration lacks a `quantization_config`, it should default to using `BitsAndBytesConfig` as it does currently.
