# Epic 5 — Chat Formatting

> **Runs on:** Local machine
> **Depends on:** Epic 4 (Trajectory Synthesis)
> **Estimated effort:** ~1.5 hours
> **Goal:** Implement `ChatFormatter` that converts `Trajectory` objects into Gemma 4 chat template formatted strings with `<start_of_turn>`, `<|tool_call|>`, `<|thought|>` tokens.

---

## Pre-Requisites

- Epic 4 is complete
- Activate environment: `source ~/python_envs/p313_llm/bin/activate`

---

## Design Reference

- `high_level/03_data_flow_pipeline.md` § 4 — Chat Template Structure
- `detailed/01_data_preprocessing.md` § 1 (ChatFormatter class diagram)

---

## Task 5.1: Create `src/data/chat_formatter.py`

**Class:** `ChatFormatter`

**Constructor:**
```python
def __init__(self, token_counter: TokenCounter, max_tokens: int = 28672) -> None:
```

**Class-level constants:**
- `_BOS_TOKEN: str = "<bos>"`
- `_START_OF_TURN: str = "<start_of_turn>"`
- `_END_OF_TURN: str = "<end_of_turn>"`
- `_TOOL_CALL_OPEN: str = "<|tool_call|>"`
- `_TOOL_CALL_CLOSE: str = "<|/tool_call|>"`
- `_THOUGHT_OPEN: str = "<|thought|>"`
- `_THOUGHT_CLOSE: str = "<|/thought|>"`
- `_USER_ROLE: str = "user"`
- `_MODEL_ROLE: str = "model"`

**Public methods:**
- `format_trajectory(trajectory: Trajectory) -> str` — convert full trajectory to formatted chat text
- `format_messages(turns: list[Turn]) -> list[dict[str, str]]` — convert to list of `{"role": ..., "content": ...}` dicts (HuggingFace conversation format)
- `count_tokens(text: str) -> int` — delegate to TokenCounter

**Private methods:**
- `_format_user_turn(turn: Turn) -> str` — format user turn (problem statement or tool result)
- `_format_model_turn(turn: Turn) -> str` — format model turn with thought blocks and tool calls
- `_format_tool_call(tool_call: ToolCall) -> str` — format a single `<|tool_call|>` JSON block
- `_format_tool_result(result: str) -> str` — format tool result as user turn content
- `_apply_chat_template(turns: list[Turn]) -> str` — assemble all turns with BOS + start/end tokens
- `_format_thought_block(thought: str) -> str` — wrap in `<|thought|>` tags

**Output format example (from design doc):**
```
<bos><start_of_turn>user
{problem statement + budget + rules}
<end_of_turn>
<start_of_turn>model
<|thought|>
{reasoning}
<|/thought|>
{text response}
<|tool_call|>
{"tool_name": "search_similar_code", "args": {"query": "...", "k": 5}}
<|/tool_call|>
<end_of_turn>
<start_of_turn>user
{"status": "ok", "results": [...]}
<end_of_turn>
...
```

---

## Task 5.2: Write Tests

### `tests/unit/data/test_chat_formatter.py`
- `test_format_trajectory_contains_bos_token`
- `test_format_trajectory_contains_start_end_turn_tokens`
- `test_format_trajectory_wraps_thought_in_tags`
- `test_format_trajectory_wraps_tool_call_in_tags`
- `test_format_model_turn_with_thought_and_tool_call`
- `test_format_model_turn_without_thought`
- `test_format_user_turn_with_tool_result`
- `test_format_user_turn_with_text`
- `test_format_tool_call_produces_valid_json`
- `test_format_messages_returns_list_of_dicts`
- `test_format_messages_alternates_roles`
- `test_count_tokens_positive` — verify returns > 0 for non-empty text

**Use `MockDataFactory` to create a mock task, synthesise a trajectory, then format it.**

---

## Task 5.3: Run CI

```bash
source ~/python_envs/p313_llm/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

ruff check src/ tests/
mypy src/
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Completion Criteria

- [ ] `ChatFormatter` produces correctly formatted chat strings
- [ ] All Gemma 4 special tokens are present
- [ ] Tool calls are valid JSON within `<|tool_call|>` tags
- [ ] Thought blocks are wrapped in `<|thought|>` tags
- [ ] Turns alternate between user and model
- [ ] `format_messages()` produces HuggingFace-compatible conversation dicts
- [ ] Tests pass with ≥90% coverage
- [ ] `ruff check` clean, `mypy` clean

---

## Files Created in This Epic

```
src/data/chat_formatter.py
tests/unit/data/test_chat_formatter.py
```
