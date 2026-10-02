# Patch Formatting and Editing Guidelines

Follow these guidelines when modifying source code in the repository.

## 1. Edit Strategy
- Always use `edit_file` for targeted modifications rather than rewriting entire files.
- Keep `old_string` small and exact (typically 3 to 10 lines) to ensure unique, reliable matching.
- Include 1-2 lines of unchanged surrounding context above and below the change to prevent ambiguity.
- If multiple hunks need changing across a file or multiple files, apply them one hunk at a time.

## 2. Forbidden Files
- **NEVER** modify test files (`test_*.py`, `*_test.py`, `tests/**`, `conftest.py`).
- **NEVER** modify build or test configuration (`pytest.ini`, `setup.cfg`, `tox.ini`, `pyproject.toml`).
- All changes must be strictly limited to production library code.

## 3. Iteration and Verification
- Verify each edit immediately after applying it.
- If `edit_file` fails due to multiple matches, re-read the file around the target line and include more context lines in `old_string`.
- If `edit_file` fails due to no match, read the file to check its current contents (earlier edits may have shifted lines).
- Execute targeted regression tests via `run_command` after each edit.

## 4. Final Submission
- Once all necessary edits are verified and the issue is resolved, invoke `submit_patch()`.
- Do not run unnecessary commands or make redundant edits after tests pass.
