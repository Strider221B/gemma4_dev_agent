We need to participate and win the Kaggle competition: 
https://www.kaggle.com/competitions/gemma-4-developer-agent

---

### 1. COMPETITION CONTEXT & CHALLENGE
The objective is to post-train (via Supervised Fine-Tuning and Reinforcement Learning) a small local model to act as an autonomous software engineering agent. The agent must independently navigate large repositories, fix bugs, and draft patches without running untrusted Python entry points on the host system.

The evaluation engine relies on a declarative-only security model. We DO NOT submit raw Python agent code. The environment compiles a declarative schema (`agent.yaml`) into a closed host registry tree (Google ADK `BaseAgent`). Our competitive edge relies entirely on post-training the model to native tool-calling perfection within this framework.

Competition Reference Assets:
- Competition details: `./documentation/competition_details`
- Kaggle CLI API queries & sample commands: `./notes.md`
- Initial dataset files: `./data`
- Dataset readme: `./data/HARNESS_README.md`
- Full file list: `./data/files.csv`
- Baseline sample solutions: `./documentation/sample_code`

---

### 2. CORE SYSTEM ARCHITECTURE & COMPUTE CONSTRAINTS

1. Compute Limitations & Local Development:
   - Remote Node: 4x NVIDIA L4 GPUs (96 GB Total VRAM). The evaluation backend runs a sharded vLLM engine (tensor_parallel_size = 4) serving the base model variant: `gemma-4-31b-it-qat-w4a16-ct`.
   - Local Machine: Very low specs. We cannot download the full training datasets or run model training locally.
   - Requirement: The data preprocessing pipeline must include a lightweight "Mock/Sample Mode" using a tiny, 2-file dummy workspace to run syntax, tool-call formatting, and pipeline structural integrity tests locally without causing memory or storage OOMs.

2. Code Architecture & Staging Contract:
   - No single-notebook monoliths. All logic must be developed as a modular, decoupled Python package following: `./documentation/prompts/coding_standards.md`
   - A final Jupyter notebook will be created whose ONLY job is to import our modular `src` pipeline entry points and execute them on Kaggle's infrastructure.
   - Clean deployment layout under `./kaggle_staging`:
     ├── agent.yaml          # Root compiled agent tree schema
     ├── eval_config.yaml    # Per-task execution budget overrides
     ├── prompts/            # System/Instruction templates included via sandboxed !include
     └── adapters/           # Fine-tuned PEFT LoRA adapters (max 8, max rank 128)

3. Submission Ceilings & Weight Enforcements:
   - Primary Constraint: Total unpacked submission size must be strictly under 3 GiB (3,221,225,472 bytes), including all adapters.
   - File Serialization: Standard PyTorch binary weights (.pt, .bin) are explicitly REJECTED by the validator. LoRA adapters MUST be saved exclusively in the `.safetensors` format.
   - Context Boundaries: Max token limit is 32,768 (combined prompt + thinking budget + generation output).

---

### 3. VALIDATION, OVERFITTING & PEER SOLUTIONS STRATEGY

1. Local Cross-Validation (CV) Framework:
   - The only feedback from the Kaggle platform is a single public metric score (Resolution Rate).
   - Design a strict, deterministic local Cross-Validation (CV) splitting strategy (e.g., Stratified K-Fold or Group K-Fold grouped by repository type/complexity) within our dataset preparation code. 
   - This framework must simulate Phase 1 (agent trajectory execution) and Phase 2 (pytest/JUnit evaluation) to yield a reliable local CV score for model checkpoints before pushing.

2. Combating Public Leaderboard Overfitting:
   - The system must explicitly track and log local validation performance versus public leaderboard scores to detect generalization decay on unseen data.
   - Post-training pipelines must incorporate strict validation early-stopping, token length regularization, and diverse trajectory formatting to ensure the agent generalises to unknown software bugs.

3. Adversarial High-Scorer Evaluation:
   - Peer solutions will be continuously captured and stored in `./documentation/sample_code/high_scorers`.
   - The design must outline a strict code inspection and adversarial validation workflow to analyze if these public high-scoring notebooks are overfitted to the public leaderboard subset. We must mathematically/empirically evaluate whether integrating their logic is robust or if it will cause a catastrophic failure on the private test dataset.

---

### 4. TECHNICAL SPECIFICATIONS & HARNESS INTERFACES

Your design must natively integrate with the 9 built-in `@budget_gated` tools exposed by `SwegemmaContext`:
1. `run_command(command)`: Capped at 300s timeouts and 5,000 characters of stdout/stderr truncation.
2. `submit_patch()`: Generates the binary git diff against the baseline. Does not count toward tool budgets. Terminates the agent turn loop immediately when a text response finishes.
3. `get_status()`: Cost-free live budget query tool.
4. `read_file(filepath, start_line, end_line)`: Capped at a dual window of 150 lines and 10,000 characters.
5. `edit_file(filepath, old_string, new_string)`: 3-tier matching engine (Exact -> Flexible Whitespace -> Tokenized Delimiter Regex). Multi-match edits without `allow_multiple=True` fail.
6. `write_file(filepath, content)`: Overwrites or creates workspace files.
7. `get_code_neighbors(node, edge_type)`: AST/Dependency graph explorer.
8. `search_similar_code(query, k)`: Cosine similarity vector search over embedded nodes.
9. `get_code_subgraph(nodes)`: Subgraph topology extraction.

Anti-Tampering & Testing Guardrails:
- The Phase 2 verification container forcefully discards any agent modifications made to test suites (e.g., `test_*.py`) or test runner configurations (`pytest.ini`, `conftest.py`, `pyproject.toml`) by running a hard `git checkout HEAD` and `git clean -f` before testing. The agent must resolve the underlying codebase bugs under `/workspace`, never the tests themselves.
- Diffs are calculated using a baseline snapshot. Any scratch verification scripts written by the agent inside `/workspace` will leak into the patch and break grading. The agent must be trained to construct scratch code exclusively inside `/tmp`.

---

### 5. RECOMMENDATION FRAMEWORK & MODELING STRATEGY

- Core Stack: Hugging Face TRL (Transformer Reinforcement Learning) + Unsloth (for fast, memory-optimized QLoRA modeling on 31B limits).
- Phase 1: Supervised Fine-Tuning (SFT) to teach the model how to cleanly generate the chat template token syntax (`<|thought|>` and `<|tool_call|>`) without experiencing token truncation mid-generation.
- Phase 2: Reinforcement Learning (GRPO / DPO) to maximize the agent's code navigation performance, correct context compaction/summarization, and local resolution metrics.
- Telemetry & Versioning: Centralized telemetry must run during Kaggle remote execution, dumping epoch, loss, reward mean, and token count performance profiles to `./logs/run_[version].log`.

---

### 6. TASK
Come up with a highly detailed architectural design to tackle this problem and generate the comprehensive documentation at: `./documentation/design`

The design must explicitly map out the following:
1. Complete Directory Tree Structure: Delineating responsibilities across data pipelines, training layers, mock sandboxes, and verification simulators.
2. Data Preprocessing & Formatter Design: Standardizing repository text blocks into target trajectory datasets.
3. Training & Validation Implementations: Detailed pseudocode configurations for the `sft_train.py`, `rl_train.py`, and the local `cv_evaluator.py` pipelines utilizing Unsloth, TRL, and group-based validation splits.
4. Overfitting & Peer Solution Merging Protocol: A step-by-step logic framework on how public solutions are stress-tested against our CV engine before logic adoption.
5. Deployment Automation Strategy: An orchestrator concept to safely package, lint, verify constraints, increment versions automatically (`kaggle datasets version -p . -m "vX.Y.Z"`), and push updates using the Kaggle CLI API.

### Local Python Env:
Python env:
source ~/python_envs/p313_llm/bin/activate
