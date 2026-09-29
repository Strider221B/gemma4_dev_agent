# SweGemma Agent — Design Documentation Index

> **Competition:** [Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent)  
> **Objective:** Post-train `gemma-4-31b-it-qat-w4a16-ct` as an autonomous SWE agent  
> **Metric:** Resolution Rate (% of tasks where agent patch passes all tests)

---

## High-Level Design

| # | Document | Description |
|---|---|---|
| 01 | [System Overview](high_level/01_system_overview.md) | Mission, competitive thesis, component inventory, constraints |
| 02 | [Architecture Diagram](high_level/02_architecture_diagram.md) | End-to-end Mermaid diagrams, complete directory tree, layer responsibilities |
| 03 | [Data Flow Pipeline](high_level/03_data_flow_pipeline.md) | 4-stage pipeline: ingestion → trajectory synthesis → formatting → dataset assembly |
| 04 | [Training Strategy](high_level/04_training_strategy.md) | Two-phase training: SFT (QLoRA) → RL (GRPO/DPO), adapter sizing, curriculum |
| 05 | [Evaluation Framework](high_level/05_evaluation_framework.md) | Group K-Fold CV, Phase 1/2 simulation, mock sandbox, overfitting detection |
| 06 | [Agent Architecture](high_level/06_agent_architecture.md) | Runtime agent YAML tree, system prompts, tool usage strategy, context management |
| 07 | [Deployment Strategy](high_level/07_deployment_strategy.md) | SubmissionPackager, ConstraintValidator, VersionManager, CI/CD pipeline |
| 08 | [Risk Mitigation](high_level/08_risk_mitigation.md) | Risk register (12 risks), multi-layer mitigations, failure recovery playbook |

## Detailed Design

| # | Document | Description |
|---|---|---|
| 01 | [Data Preprocessing](detailed/01_data_preprocessing.md) | Class diagrams, data models, trajectory synthesis algorithm, patch parser, mock mode |
| 02 | [SFT Training](detailed/02_sft_training.md) | Full SFT pipeline: Unsloth loading, LoRA config, TRL SFTTrainer, curriculum, telemetry |
| 03 | [RL Training](detailed/03_rl_training.md) | GRPO/DPO implementations, multi-signal reward model, preference pair generation |
| 04 | [CV Evaluator](detailed/04_cv_evaluator.md) | Full CV pipeline, budget simulation, report generation, actionable recommendations |
| 05 | [Peer Solution Protocol](detailed/05_peer_solution_protocol.md) | 4-step adversarial validation, leakage detection, bootstrap significance, technique isolation |

---

## Quick Reference

### Key Constraints
- **Model**: `gemma-4-31b-it-qat-w4a16-ct` (single model, mandatory)
- **Submission size**: < 3 GiB total (including adapters)
- **Adapter format**: `.safetensors` only (no `.bin`, `.pt`)
- **Max adapters**: 8, max rank 128
- **Context window**: 32,768 tokens
- **Eval time**: 12 hours for all tasks

### Training Stack
- **Unsloth** — Memory-optimised 4-bit model loading
- **TRL** — SFTTrainer, GRPOTrainer, DPOTrainer
- **PEFT** — LoRA adapter management
- **HuggingFace Datasets** — Data pipeline

### Agent Tools (9 total)
| Tool | Budget-Gated | Key Limits |
|---|---|---|
| `run_command` | ✅ | 300s timeout, 5000 char output |
| `submit_patch` | ❌ Free | Terminates agent loop |
| `get_status` | ❌ Free | Returns budget info |
| `read_file` | ✅ | 150 lines, 10000 chars |
| `edit_file` | ✅ | 3-tier matching, uniqueness check |
| `write_file` | ✅ | Creates parent dirs |
| `get_code_neighbors` | ✅ | 50 max neighbors, 4-tier resolution |
| `search_similar_code` | ✅ | Cosine similarity, symbol-based query |
| `get_code_subgraph` | ✅ | Induced subgraph extraction |

### Critical Rules
1. Never modify test files — they are reset in Phase 2
2. Write scratch scripts to `/tmp/`, never `/workspace/`
3. Save adapters as `.safetensors` exclusively
4. Train to emit complete `<|tool_call|>` blocks without truncation
5. Use `AgentTool(skip_summarization=true)` to isolate context cost
