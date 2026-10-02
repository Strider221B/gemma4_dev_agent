"""Unit tests for PromptCompiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.deployment.prompt_compiler import PromptCompiler


class TestPromptCompiler:
    """Test suite for PromptCompiler template and include resolution."""

    _SYSTEM_FILE: str = "prompts/system.md"
    _ROLE_VAR: str = "role"
    _NAME_VAR: str = "name"
    _PROMPT_TEMPLATE: str = "Hello {name}, your role is {role}."
    _COMPILED_RESULT: str = "Hello Alice, your role is coder."
    _PROMPT_CONTENT: str = "You are an expert autonomous software engineer."

    @pytest.fixture
    def templates_dir(self, tmp_path: Path) -> Path:
        """Create a templates directory fixture with sample prompt files."""
        root = tmp_path / "templates"
        prompts = root / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        (root / self._SYSTEM_FILE).write_text(self._PROMPT_CONTENT, encoding="utf-8")
        return root

    def test_compile_prompt_substitutes_variables(self, templates_dir: Path) -> None:
        """Verify variable placeholders are correctly replaced with provided values."""
        template_path = templates_dir / "template.txt"
        template_path.write_text(self._PROMPT_TEMPLATE, encoding="utf-8")
        compiler = PromptCompiler(str(templates_dir))
        result = compiler.compile_prompt(
            str(template_path),
            {self._NAME_VAR: "Alice", self._ROLE_VAR: "coder"},
        )
        assert result == self._COMPILED_RESULT

    def test_resolve_includes_replaces_directives(self, templates_dir: Path) -> None:
        """Verify !include directives are replaced with referenced file contents."""
        yaml_content = f"instruction: !include {self._SYSTEM_FILE}\n"
        compiler = PromptCompiler(str(templates_dir))
        resolved = compiler.resolve_includes(yaml_content)
        assert f"instruction: {self._PROMPT_CONTENT}\n" == resolved

    def test_resolve_includes_traversal_raises_value_error(self, templates_dir: Path) -> None:
        """Verify path traversal in !include directive triggers ValueError."""
        yaml_content = "instruction: !include ../../outside.txt\n"
        compiler = PromptCompiler(str(templates_dir))
        with pytest.raises(ValueError, match="Path traversal detected"):
            compiler.resolve_includes(yaml_content)

    def test_resolve_includes_missing_file_raises_error(self, templates_dir: Path) -> None:
        """Verify !include referencing non-existent file raises FileNotFoundError."""
        yaml_content = "instruction: !include nonexistent.txt\n"
        compiler = PromptCompiler(str(templates_dir))
        with pytest.raises(FileNotFoundError, match="Included file does not exist"):
            compiler.resolve_includes(yaml_content)

    def test_compile_prompt_missing_template_raises_error(self, templates_dir: Path) -> None:
        """Verify compiling a non-existent template raises FileNotFoundError."""
        compiler = PromptCompiler(str(templates_dir))
        with pytest.raises(FileNotFoundError, match="Template not found"):
            compiler.compile_prompt("missing_template.txt", {})
