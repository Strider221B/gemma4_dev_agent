# 08 — Risk Mitigation

---

## 1. Risk Register

| ID | Risk | Likelihood | Impact | Severity | Mitigation |
|---|---|---|---|---|---|
| R1 | **Token truncation** during tool-call generation | High | High | **Critical** | Set `thinking_budget: 4096`, train on incremental edits, enforce `max_seq_length: 28672` |
| R2 | **Public LB overfitting** — CV score diverges from LB | High | Critical | **Critical** | Group K-Fold by repo, MetricTracker with gap alerts, early stopping |
| R3 | **Test set uses unknown repos** — model struggles with unseen code | High | High | **Critical** | Train on diverse trajectories, avoid repo-specific patterns, RL generalisation |
| R4 | **Adapter size exceeds 3 GiB** | Low | Critical | **High** | Pre-flight size validation, conservative rank (r=32), size monitoring |
| R5 | **edit_file matching failures** at eval time | Medium | High | **High** | Train on 3-tier matching patterns, use focused old_string (5-10 lines) |
| R6 | **Scratch files leak into patch** | Medium | High | **High** | Train to use `/tmp/` exclusively, post-edit cleanup verification |
| R7 | **OOM on 4×L4 during training** | Medium | High | **High** | Gradient checkpointing, batch_size=1 + accumulation, Unsloth memory optimisation |
| R8 | **Peer solution is LB-overfit** — adopting it tanks private score | Medium | High | **High** | Adversarial CV stress test before any adoption |
| R9 | **Agent modifies test files** — changes discarded in Phase 2 | Medium | Medium | **Medium** | Explicit system prompt rules, trajectory filtering, reward penalty |
| R10 | **Agent exhausts tool budget** without submitting | Medium | Medium | **Medium** | Budget awareness training, `get_status()` calls, auto-submit on low budget |
| R11 | **Kaggle dataset push fails** | Low | Medium | **Medium** | Pre-flight CLI dry-run, version rollback capability |
| R12 | **vLLM inference errors** (OOM, timeout) | Low | High | **Medium** | Conservative `gpu_memory_utilization: 0.80`, retry plugin (5 retries) |

---

## 2. Critical Risk Mitigations

### R1: Token Truncation

The most common failure mode. When the model's `<|tool_call|>` block is truncated:
1. The harness sends a nudge: *"emit your next tool call immediately"*
2. After 3 nudges without tool calls, the session terminates
3. No patch is submitted → `resolved = False`

**Multi-Layer Defence**:
```
Layer 1 (Training):
  - Cap training trajectories at 28K tokens
  - Include truncation-recovery trajectories in SFT data
  - Penalise long thinking blocks in RL reward

Layer 2 (Prompts):
  - System prompt: "keep reasoning under a few sentences"
  - System prompt: "split large edits into smaller operations"

Layer 3 (Architecture):
  - thinking_budget: 4096 (limits internal reasoning tokens)
  - max_output_tokens: 16384 (generous but bounded)
  - AgentTool delegation isolates navigation token cost

Layer 4 (Runtime):
  - ADK EventsCompaction at 14,336 token threshold
  - Harness nudge mechanism (3 chances to recover)
```

### R2: Public LB Overfitting

```
Detection Protocol:
  1. Maintain MetricTracker history: (version, cv_score, lb_score)
  2. Compute generalization gap: |cv_score - lb_score|
  3. If gap > 0.15 OR (cv↑ while lb↓) → STRONG overfitting signal

Prevention:
  1. Group K-Fold by repo (never train+validate on same repo)
  2. Include trajectory augmentation (tool order permutation, error injection)
  3. Token length regularisation (penalise >24K token trajectories)
  4. Early stopping on eval_loss, not resolution_rate
```

### R3: Unknown Test Repos

The test set uses **private repositories** — the model has never seen their code structure, naming conventions, or build systems.

```
Generalisation Strategies:
  1. Group K-Fold: each fold holds out entire repo → forces cross-repo transfer
  2. Strip repo-specific identifiers from training trajectories
  3. Focus on general bug-fixing patterns, not repo-specific APIs
  4. Leverage code graph tools (search_similar_code, get_code_neighbors)
     to adapt to unfamiliar codebases at runtime
  5. System prompt emphasises "inspect existing conventions first"
```

---

## 3. Failure Recovery Playbook

### 3.1 Training Failures

| Failure | Detection | Recovery |
|---|---|---|
| OOM during SFT | CUDA OOM error | Reduce `max_seq_length`, enable gradient checkpointing, reduce batch size |
| NaN loss | Loss becomes NaN | Reduce learning rate, increase warmup, check data for extreme token lengths |
| Adapter too large | Size > 1.5 GiB | Reduce LoRA rank from 32 → 16, remove less critical target_modules |
| Training divergence | Eval loss increases for >3 eval periods | Revert to previous checkpoint, reduce LR by 50% |

### 3.2 Evaluation Failures

| Failure | Detection | Recovery |
|---|---|---|
| CV score drops after RL | CV < SFT baseline | Discard RL checkpoint, use SFT-only adapter |
| High truncation rate (>5%) | Trajectory analysis | Reduce thinking_budget, add more incremental-edit training data |
| Low tool efficiency (>60 calls avg) | CV metrics | Add efficiency bonus in RL reward, train shorter trajectories |

### 3.3 Deployment Failures

| Failure | Detection | Recovery |
|---|---|---|
| Kaggle push fails | CLI error | Check dataset-metadata.json, retry with `--force` |
| Submission validation fails | Pre-flight checks | Fix constraint violations, re-run packager |
| Runtime OOM on 4×L4 | vLLM crash | Reduce `max_model_len`, lower `gpu_memory_utilization` |

---

## 4. Monitoring & Alerting

### 4.1 Training Telemetry Alerts

```python
class TelemetryAlerts:
    _MAX_LOSS_INCREASE_RATIO: float = 1.5
    _MIN_TOKENS_PER_SECOND: int = 500
    _MAX_GPU_MEMORY_RATIO: float = 0.95
    
    def check(self, metrics: TrainingMetrics) -> list[Alert]:
        alerts = []
        
        if metrics.eval_loss > metrics.prev_eval_loss * self._MAX_LOSS_INCREASE_RATIO:
            alerts.append(Alert("LOSS_SPIKE", "Eval loss increased by >50%"))
        
        if metrics.tokens_per_second < self._MIN_TOKENS_PER_SECOND:
            alerts.append(Alert("SLOW_TRAINING", "Token throughput below threshold"))
        
        if max(metrics.gpu_memory_ratios) > self._MAX_GPU_MEMORY_RATIO:
            alerts.append(Alert("GPU_MEMORY_PRESSURE", "GPU memory >95% utilised"))
        
        return alerts
```

### 4.2 Competition Timeline Strategy

| Week | Focus | Deliverable |
|---|---|---|
| Week 1 | Data pipeline + SFT baseline | `sft_lora` adapter, CV score baseline |
| Week 2 | Prompt engineering + agent architecture tuning | Optimised `agent.yaml`, improved CV |
| Week 3 | RL training (GRPO) | `rl_lora` adapter, CV improvement |
| Week 4 | Peer solution analysis + ensemble strategies | Final submission, adversarial validation |
| Final days | Conservative — only submit if CV↑ AND gap stable | Stable final version |
