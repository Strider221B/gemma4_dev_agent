# Adspire Coding Standards

> **This document is the single source of truth for all coding standards on the Adspire project.**
> Every AI agent, prompt, and developer MUST adhere to these rules — regardless of which prompt template or workflow they use.
> If your prompt does not explicitly mention a rule below, the rule still applies.

---

## 1. SOLID Principles

Every class must follow all five SOLID principles:

- **S — Single Responsibility**: Each class must have ONE and only one reason to change. Do not add unrelated responsibilities.
- **O — Open/Closed**: Design for extension. Classes should be open for extension, closed for modification. Use abstract methods or hooks where appropriate.
- **L — Liskov Substitution**: Subclasses must be substitutable for their base classes without breaking behavior.
- **I — Interface Segregation**: Implement only the interface methods that the class needs. No client should be forced to depend on methods it does not use.
- **D — Dependency Inversion**: Depend on abstractions (interfaces/abstract classes), not concrete implementations. Accept dependencies via constructor injection.

---

## 2. DRY (Don't Repeat Yourself)

- No duplicated logic. If a pattern is repeated, extract it into a private method, base class, utility class, or shared module.
- Reuse base class functionality where applicable.

---

## 3. YAGNI (You Aren't Gonna Need It)

- Only implement what is required by the current task/epic. No speculative features, no "nice-to-have" extras, no "future-proofing" code.
- No TODO comments suggesting future work.

---

## 4. Strict OOP

- ALL functions and members MUST be inside classes. No free-standing functions. No module-level executable code.
- Only `imports`, `typing` definitions, and `constants` may exist outside the class.
- Entry points (e.g., `main.py`) must use a class with a `run()` or `main()` method, invoked under `if __name__ == "__main__":`.

---

## 5. Member Ordering (STRICTLY ENFORCED)

Within every class, arrange members in this exact order:

1. Class-level variables / constants
2. `__init__()`
3. Public methods (alphabetical or logical grouping)
4. Protected methods (prefix: `_`) (alphabetical or logical grouping)
5. Private methods (prefix: `__`) (alphabetical or logical grouping)

Public members/methods MUST appear BEFORE non-public (protected/private) members/methods.

---

## 6. Method Length Limit — ≤ 20 Lines

- Count ONLY the logical lines of code in the method body. If a long line of code is split into
  multiple lines, count it as one.
- Do NOT count: the `def` line, decorators, docstrings, blank lines, comments within the method.
- If a method's logic requires more than 20 lines, it MUST be split into smaller private helper methods.
- This is a **HARD LIMIT**. No exceptions.

---

## 7. File Length Limit — ≤ 500 Lines

- The entire file (including imports, comments, blank lines, everything) must not exceed 500 lines.
- If approaching the limit, split the class or rethink the design.

---

## 8. One Class Per File

- Each file contains exactly ONE class.
- The file name must match the class name (casing differences allowed, e.g., `UserService` → `user_service.py`).
- Helper/inner classes are NOT allowed. Extract them into separate files.

---

## 9. Testing Standards — 90%+ Coverage

- Use `pytest` as the testing framework.
- Use `unittest.mock` (or `pytest-mock`) for mocking dependencies.
- Every public method must have unit tests. Every branch/condition must be tested.
- Each test method tests ONE behavior/scenario.
- Test method naming: `test_<method_name>_<scenario_description>`
- Include:
  - ✅ Happy path tests
  - ✅ Edge case tests
  - ✅ Error/exception tests
  - ✅ Boundary condition tests
- Use `@pytest.fixture` for shared setup.
- Mock ALL external dependencies (databases, APIs, file systems, etc.).
- Assert on specific values, not just "no exception thrown."
- Aim for 100% line and branch coverage of the class under test.
- Each test file corresponds to a source file (e.g., `user_service.py` → `test_user_service.py`).

---

## 10. No Magic Strings or Numbers

- All literal strings and numeric values used in logic (status codes, config keys, thresholds, default values, error messages, etc.) MUST be defined as named constants.
- If the constant is specific to a single class, define it as a **non-public class-level constant** (use a leading underscore, e.g., `_MAX_RETRIES = 3`). Class-level constants MUST NOT be public — they are internal implementation details.
- If the same constant is used across multiple files, extract it into a **dedicated constants file** (e.g., `constants.py`) in the appropriate module.
- Raw string/number literals in method bodies (e.g., `if status == 200`, `role = "admin"`) are **NOT acceptable**.
- Acceptable outside-class constants: only those at module level that are true shared constants (imported by other modules).

---

## 11. Python Virtual Environment

- Do NOT create a new Python virtual environment unless one does not already exist.
- Activate with: `source ~/python_envs/p313_llm/bin/activate`.
- All dependency installations (`pip install`, `uv sync`) and script executions MUST run within the active virtual environment.

---

## 12. Code Quality Standards

- Use **type hints** on ALL method parameters and return types.
- Write **docstrings** for every class and every public method (Google style or NumPy style — be consistent).
- Use **meaningful variable names** (no single letters except loop counters like `i`, `j`).
- Handle errors explicitly — use custom exceptions where the plan specifies, otherwise use appropriate built-in exceptions.
- Use `from __future__ import annotations` if needed for forward references.
- Follow **PEP 8** formatting.

---

## 13. High-Level Design Adherence

All work MUST be consistent with the high-level design documents located at `documentation/design/high_level/`. These documents represent agreed-upon architectural decisions and are the **source of truth** for the system's design:

- `01_system_overview.md` — System overview and scope
- `02_architecture_diagram.md` — Architecture and component layout
- `03_data_flow_pipeline.md` — Data flow and pipeline design
- `04_data_model.md` — Data model and schema definitions
- `05_tech_stack.md` — Technology stack decisions
- `06_api_design.md` — API design and contracts
- `07_sequence_diagrams.md` — Interaction and sequence diagrams
- `08_deployment_architecture.md` — Deployment architecture
- `09_non_functional_requirements.md` — Non-functional requirements (performance, security, etc.)

If any implementation contradicts these documents, the conflict MUST be flagged and the deviation justified explicitly.

---

## 14. Local CI Verification (MANDATORY)

Before considering any code complete, you MUST run the full CI pipeline locally and ensure all checks pass. The CI is configured in `pyproject.toml` and consists of three stages that mirror the GitHub Actions pipeline. **Code that fails any of these checks will break the CI build — do NOT commit or submit code that has not passed all three stages.**

### Stage 1: Lint (ruff)
```bash
# Activate your virtual environment first (see §11 for your machine's path)
ruff check ./src
```
- Configuration: `[tool.ruff]` and `[tool.ruff.lint]` in `pyproject.toml`
- Rules: `E` (pycodestyle errors), `W` (pycodestyle warnings), `F` (pyflakes), `I` (isort)
- Line length: 100 characters
- **Zero lint errors allowed**. Fix all violations before proceeding.

### Stage 2: Type Check (mypy)
```bash
# Activate your virtual environment first (see §11 for your machine's path)
mypy ./src
```
- Configuration: `[tool.mypy]` in `pyproject.toml`
- Mode: `strict = true`
- Plugins: `pydantic.mypy`
- **Zero type errors allowed**. Fix all violations before proceeding.

### Stage 3: Tests with Coverage (pytest)
```bash
# Activate your virtual environment first (see §11 for your machine's path)
uv run pytest
```
- Configuration: `[tool.pytest.ini_options]` and `[tool.coverage.*]` in `pyproject.toml`
- Runs with: `--cov=app --cov-report=term-missing --cov-report=xml`
- Coverage source: `app/` with branch coverage enabled
- **Minimum coverage threshold: 90%** (`fail_under = 90`). The build FAILS if coverage drops below this.
- **Zero test failures allowed**.

### If a check fails:
1. Fix the issue in the source code or test code.
2. Re-run the failing stage.
3. Repeat until all three stages pass cleanly.
4. Only THEN mark the task as complete.

---

## Verification Checklist

Before submitting any code, verify:

| Check | Status |
|---|---|
| Single class in file | YES/NO |
| File name matches class name | YES/NO |
| Public members before non-public | YES/NO |
| All methods ≤ 20 lines (body only, excluding comments) | YES/NO |
| File ≤ 500 lines | YES/NO (actual count: ___) |
| Type hints on all parameters and returns | YES/NO |
| Docstrings on class and public methods | YES/NO |
| No free-standing functions | YES/NO |
| Dependencies injected via constructor | YES/NO |
| SOLID compliance | YES/NO |
| DRY compliance | YES/NO |
| YAGNI compliance | YES/NO |
| No magic strings/numbers | YES/NO |
| Local CI passes (ruff + mypy + pytest) | YES/NO |
