# Epic 12 — Submission Templates

> **Runs on:** Local machine
> **Depends on:** Epic 11 (Deployment Pipeline)
> **Estimated effort:** ~1.5 hours
> **Goal:** Create all agent YAML configurations, system/navigator prompts, skills, and sub-agent definitions that form the runtime agent architecture.

---

## Pre-Requisites

- Epic 11 is complete
- Activate environment: `source ~/python_envs/p313_llm/bin/activate`

---

## Design Reference

- `high_level/06_agent_architecture.md` — Agent tree, prompts, tool usage
- `high_level/07_deployment_strategy.md` § 4.2 — Kaggle file layout

---

## Task 12.1: Create Root Agent YAML

**File:** `kaggle_staging/submission_templates/agent.yaml`

Content from `high_level/06_agent_architecture.md` § 2.1:

```yaml
name: root_coder
model: gemma-4-31b-it-qat-w4a16-ct
adapter: coder_lora

instruction: !include prompts/system.md

tools:
  - edit_file
  - write_file
  - run_command
  - submit_patch
  - get_status
  - agent_tool:
      config_path: sub_agents/code_analyzer.yaml
      skip_summarization: true

generate_content_config:
  max_output_tokens: 16384
  temperature: 0.2
  top_p: 0.95
  top_k: 40
  thinking_config:
    include_thoughts: true
    thinking_budget: 4096
```

---

## Task 12.2: Create Code Analyzer Sub-Agent YAML

**File:** `kaggle_staging/submission_templates/sub_agents/code_analyzer.yaml`

Content from `high_level/06_agent_architecture.md` § 2.2.

---

## Task 12.3: Create Eval Config YAML

**File:** `kaggle_staging/submission_templates/eval_config.yaml`

```yaml
evaluation:
  max_time_minutes: 30
  max_tool_calls: 75
  max_turns: 200
  timeout_seconds: 300
```

---

## Task 12.4: Create System Prompt

**File:** `kaggle_staging/submission_templates/prompts/system.md`

Content from `high_level/06_agent_architecture.md` § 3 — the full system prompt for the root coder agent.

---

## Task 12.5: Create Navigator Prompt

**File:** `kaggle_staging/submission_templates/prompts/navigator.md`

Content from `high_level/06_agent_architecture.md` § 3.1.

---

## Task 12.6: Create Patch Guidelines Prompt

**File:** `kaggle_staging/submission_templates/prompts/patch_guidelines.md`

Create a concise patch formatting guide covering:
- Use `edit_file` with small, precise `old_string` (3-10 lines)
- Apply one hunk at a time
- Never modify test files
- Verify after each edit
- Use `submit_patch()` when all edits are verified

---

## Task 12.7: Create Repo Navigation Skill

**File:** `kaggle_staging/submission_templates/skills/repo_navigation/SKILL.md`

Create a skill document that teaches the agent:
- When to use `search_similar_code` vs `get_code_neighbors` vs `get_code_subgraph`
- How to navigate unfamiliar repos efficiently
- How to interpret code graph results
- When to delegate to the code_analyzer sub-agent

---

## Task 12.8: Validate Templates

Create a validation script or use `ConstraintValidator` to verify:

```bash
source ~/python_envs/p313_llm/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

# Verify all YAML files parse correctly
python -c "
import yaml
import pathlib

template_dir = pathlib.Path('kaggle_staging/submission_templates')
for yaml_file in template_dir.rglob('*.yaml'):
    with open(yaml_file) as f:
        # Use safe_load to verify syntax (won't resolve !include)
        content = f.read()
        # Basic syntax check - replace !include with placeholder
        content_clean = content.replace('!include', '#include')
        yaml.safe_load(content_clean)
        print(f'  OK: {yaml_file}')

print('All YAML files are syntactically valid.')
"
```

---

## Task 12.9: Verify File Structure

The `kaggle_staging/submission_templates/` directory should look like:

```
submission_templates/
├── agent.yaml
├── eval_config.yaml
├── prompts/
│   ├── system.md
│   ├── navigator.md
│   └── patch_guidelines.md
├── sub_agents/
│   └── code_analyzer.yaml
└── skills/
    └── repo_navigation/
        └── SKILL.md
```

---

## Completion Criteria

- [ ] `agent.yaml` exists with correct root agent config
- [ ] `code_analyzer.yaml` exists with correct sub-agent config
- [ ] `eval_config.yaml` exists with budget settings
- [ ] All prompt `.md` files exist with substantive content
- [ ] Skill `SKILL.md` exists
- [ ] All YAML files parse correctly
- [ ] Directory structure matches specification

---

## Files Created in This Epic

```
kaggle_staging/submission_templates/agent.yaml
kaggle_staging/submission_templates/eval_config.yaml
kaggle_staging/submission_templates/sub_agents/code_analyzer.yaml
kaggle_staging/submission_templates/prompts/system.md
kaggle_staging/submission_templates/prompts/navigator.md
kaggle_staging/submission_templates/prompts/patch_guidelines.md
kaggle_staging/submission_templates/skills/repo_navigation/SKILL.md
```
