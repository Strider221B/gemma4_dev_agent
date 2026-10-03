# Enhancement Design 05 — Gemma 4 ClippableLinear PEFT Support

---

## 1. Problem Statement

During the Supervised Fine-Tuning (SFT) phase of the Gemma 4 dev agent on Kaggle (and similarly during RL training), applying LoRA adapters via PEFT's `get_peft_model()` fails with a fatal `ValueError`:

```text
ValueError: Target module Gemma4ClippableLinear(
  (linear): Linear(in_features=1152, out_features=1152, bias=False)
) is not supported. Currently, only the following modules are supported: `torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, `transformers.pytorch_utils.Conv1D`, `torch.nn.MultiheadAttention.`.
```

### Context & Reproduction
Executing the SFT pipeline with the target model `google/gemma-4-31b-it-qat-w4a16-ct`:

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
```

Triggers the following call stack:
```text
/src/training/sft_trainer.py in run(self, config, dataset)
    model = self._apply_lora(model, config.lora)
/src/training/sft_trainer.py in _apply_lora(self, model, config)
    lora_model = get_peft_model(cast(Any, model), lora_config)
/peft/tuners/lora/model.py in _create_new_module(lora_config, adapter_name, target, **kwargs)
    raise ValueError(f"Target module {target} is not supported...")
```

### Root Cause Analysis

1. **Gemma 4 Architecture**:
   In Hugging Face `transformers` (v5+), the Gemma 4 architecture (`modeling_gemma4.py`) wraps projection layers in a custom module `Gemma4ClippableLinear`:
   ```python
   class Gemma4ClippableLinear(nn.Module):
       def __init__(self, config, in_features, out_features):
           super().__init__()
           self.use_clipped_linears = config.use_clipped_linears
           self.linear = nn.Linear(in_features, out_features, bias=False)
           if self.use_clipped_linears:
               self.input_min = nn.Buffer(torch.tensor(-float("inf")))
               self.input_max = nn.Buffer(torch.tensor(float("inf")))
               self.output_min = nn.Buffer(torch.tensor(-float("inf")))
               self.output_max = nn.Buffer(torch.tensor(float("inf")))

       def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
           if self.use_clipped_linears:
               hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
           hidden_states = self.linear(hidden_states)
           if self.use_clipped_linears:
               hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
           return hidden_states
   ```
   All standard projection targets configured in `configs/sft_config.yaml` (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) are instances of `Gemma4ClippableLinear`.

2. **PEFT Target Matching**:
   PEFT scans the model's named modules matching strings in `target_modules`. When matching e.g. `model.layers.0.self_attn.q_proj`, PEFT finds an instance of `Gemma4ClippableLinear`. Because `Gemma4ClippableLinear` is an `nn.Module` subclass wrapping `nn.Linear` (rather than inheriting from `nn.Linear`), PEFT does not recognize it as a supported linear module and raises `ValueError`.

3. **Critical Invariant**:
   Activation clipping (`torch.clamp`) in `Gemma4ClippableLinear` is vital for preventing numerical instability and extreme activation outliers in quantized models (such as `W4A16` QAT). Any solution must preserve this clipping wrapper rather than stripping or unwrapping it.

---

## 2. Solution Design

### 2.1 Submodule Targeting (`.linear` Target Projection)

In PyTorch module trees, each `Gemma4ClippableLinear` module has a child submodule named `linear` of type `torch.nn.Linear`:
```text
model.layers[i].self_attn.q_proj           <-- Gemma4ClippableLinear
model.layers[i].self_attn.q_proj.linear    <-- torch.nn.Linear
```

In PEFT, module resolution matches suffixes:
- If target module is `"q_proj.linear"`, PEFT matches `model.layers[i].self_attn.q_proj.linear`.
- The parent module is `q_proj` (`Gemma4ClippableLinear`).
- The target attribute is `linear` (`torch.nn.Linear`).
- PEFT replaces `q_proj.linear` with `peft.tuners.lora.Linear`.

During execution of `q_proj.forward(x)`:
1. Input is clamped: `hidden_states = torch.clamp(hidden_states, input_min, input_max)`
2. `self.linear(hidden_states)` is called: This evaluates the PEFT `LoraLinear` layer (base weights + LoRA adapter `B @ A * scaling`).
3. Output is clamped: `hidden_states = torch.clamp(hidden_states, output_min, output_max)`

This preserves full clipping invariants while attaching LoRA adapters natively without modifying PEFT internals or losing compatibility.

### 2.2 Dynamic Target Module Resolution: `LoRATargetModuleResolver`

Because the training pipelines must support:
1. Actual Gemma 4 models on Kaggle (where projections are `Gemma4ClippableLinear`),
2. Standard transformers models or test mocks (where projections may be direct `nn.Linear`),
3. Explicit user configuration in YAML or code,

we introduce a dedicated resolver class: `LoRATargetModuleResolver`.

#### Resolution Workflow

```mermaid
flowchart TD
    START["Input: Base Model & target_modules<br/>['q_proj', 'k_proj', ...]"] --> INSPECT["Inspect Model Submodules for Target Names"]
    INSPECT --> CHECK{"Are target modules instances of<br/>ClippableLinear or do they have<br/>a child .linear submodule?"}
    CHECK -->|Yes (Gemma 4)| ADAPT["Append '.linear' suffix to targets<br/>['q_proj.linear', 'k_proj.linear', ...]"]
    CHECK -->|No (Standard Model)| KEEP["Retain original targets<br/>['q_proj', 'k_proj', ...]"]
    CHECK -->|Already formatted| KEEP
    ADAPT --> RETURN["Return Resolved Target Modules to LoraConfig"]
    KEEP --> RETURN
```

---

## 3. Detailed Component Architecture

### 3.1 Class Contract: `LoRATargetModuleResolver`

Located at: `src/training/lora_target_resolver.py`

```python
class LoRATargetModuleResolver:
    """Resolves and adapts LoRA target modules for model architectures like Gemma 4."""

    _CLIPPABLE_CLASS_NAME: str = "Gemma4ClippableLinear"
    _LINEAR_ATTR: str = "linear"
    _DOT_LINEAR: str = ".linear"

    def resolve(self, model: object, target_modules: list[str]) -> list[str]:
        """Resolve target module names, appending .linear if targets are wrapped clippable layers."""
        ...
```

#### Responsibilities:
1. Traverse named modules of the base model to locate occurrences of configured target modules.
2. Check if the matched module is a wrapper containing a `.linear` child submodule (e.g. `Gemma4ClippableLinear`).
3. If wrapped, map each target module `<name>` to `<name>.linear` unless it is already suffixed with `.linear`.
4. If not wrapped or if `model` is a mock/standard `nn.Linear`, return the original `target_modules`.
5. Ensure returned target module list contains no duplicates while preserving deterministic ordering.

### 3.2 Integration into `SFTTrainerPipeline`

In `src/training/sft_trainer.py`, update `_apply_lora()`:

```python
def _apply_lora(self, model: object, config: LoRAConfig) -> object:
    """Apply QLoRA adapters targeting specified linear projection layers."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model

    from src.training.lora_target_resolver import LoRATargetModuleResolver

    resolver = LoRATargetModuleResolver()
    resolved_targets = resolver.resolve(model, config.target_modules)

    bias_val: Any = config.bias
    lora_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=resolved_targets,
        bias=bias_val,
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,
    )
    self._enable_input_grads(model)
    lora_model = get_peft_model(cast(Any, model), lora_config)
    self._enable_gradient_checkpointing(lora_model)
    self._log_trainable_params(lora_model)
    return lora_model
```

### 3.3 Integration into `RLTrainerPipeline`

In `src/training/rl_trainer.py`, update `_merge_and_reapply_lora()`:

```python
def _merge_and_reapply_lora(self, model: object, config: RLConfig) -> object:
    """Attach fresh LoRA adapter layers on top of merged SFT base model."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model

    from src.training.lora_target_resolver import LoRATargetModuleResolver

    resolver = LoRATargetModuleResolver()
    resolved_targets = resolver.resolve(model, list(self._TARGET_MODULES))

    lora_r = getattr(config.grpo, "lora_r", self._DEFAULT_LORA_R)
    lora_config = PeftLoraConfig(
        r=lora_r,
        lora_alpha=self._DEFAULT_LORA_ALPHA,
        lora_dropout=self._DEFAULT_LORA_DROPOUT,
        target_modules=resolved_targets,
        task_type=TaskType.CAUSAL_LM,
    )
    self._enable_input_grads(model)
    lora_model = get_peft_model(cast(Any, model), lora_config)
    self._enable_gradient_checkpointing(lora_model)
    return lora_model
```

### 3.4 Compatibility with `configs/sft_config.yaml` & `LoRAConfig`

The default configuration in `src/config/lora_config.py` and `configs/sft_config.yaml`:
```yaml
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```
remains intuitive and canonical. The resolver automatically handles the `.linear` expansion at runtime when targeting Gemma 4 architectures without requiring breaking changes in YAML schemas.

---

## 4. Architectural Adherence & Standards Compliance

### 4.1 SOLID Principles
- **Single Responsibility**: `LoRATargetModuleResolver` only resolves module naming discrepancies between model architectures and PEFT targets.
- **Open/Closed**: Open to supporting additional wrapped layer types by extending internal module inspection rules without modifying trainer pipeline logic.
- **Liskov Substitution**: Returns standard list of string target module names compatible with all PEFT adapters.
- **Interface Segregation**: Single clean entry point: `resolve(model, target_modules)`.
- **Dependency Inversion**: Trainers depend on the resolver abstraction rather than embedding architecture-specific string manipulations.

### 4.2 Coding Standards (Adspire)
- Strict OOP: class with class-level constants, no free-standing functions.
- All methods <= 20 lines.
- File length <= 500 lines.
- No magic strings: `_CLIPPABLE_CLASS_NAME`, `_LINEAR_ATTR`, `_DOT_LINEAR` defined as constants.
- Type annotations across all parameters and returns.

---

## 5. Risk Assessment & Verification Strategy

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Module name mismatches in PEFT | Low | High | Comprehensive unit tests for resolver matching logic against mock clippable and non-clippable modules. |
| Loss of Gemma 4 activation clipping | None | Critical | Target `.linear` replaces inner projection, leaving outer `Gemma4ClippableLinear` clamping intact. |
| Checkpoint size increase | None | Medium | Number of trainable parameters remains identical (only linear weights are adapted). |
| Compatibility with `merge_and_unload()` | Low | High | PEFT merges weights directly into the target `nn.Linear` inside `Gemma4ClippableLinear.linear`. |
