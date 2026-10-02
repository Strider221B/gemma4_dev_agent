"""Prompt compiler for YAML includes and prompt variable substitution."""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar


class PromptCompiler:
    """Compiles prompts and resolves !include directives in declarative configurations."""

    _INCLUDE_PATTERN: ClassVar[str] = r"!include\s+['\"]?([^'\"\s\n\r]+)['\"]?"
    _VAR_PREFIX: ClassVar[str] = "{"
    _VAR_SUFFIX: ClassVar[str] = "}"

    def __init__(self, templates_dir: str) -> None:
        """Initialize PromptCompiler with a templates root directory.

        Args:
            templates_dir: Base directory for resolving included files and templates.
        """
        self._templates_dir = templates_dir

    def compile_prompt(self, template_path: str, variables: dict[str, str]) -> str:
        """Read template, substitute variables, and return compiled prompt string.

        Args:
            template_path: Relative or absolute path to template file.
            variables: Dictionary of variable name to replacement value.

        Returns:
            Compiled prompt string with substitutions applied.
        """
        template = self._read_template(template_path)
        return self._substitute_variables(template, variables)

    def resolve_includes(self, yaml_content: str) -> str:
        """Replace all !include path directives with file contents.

        Args:
            yaml_content: YAML content containing !include directives.

        Returns:
            Resolved YAML content.
        """
        return re.sub(
            self._INCLUDE_PATTERN,
            self._replace_single_include,
            yaml_content,
        )

    def _read_template(self, path: str) -> str:
        target = Path(path)
        if not target.is_absolute():
            target = Path(self._templates_dir) / path
        if not target.is_file():
            raise FileNotFoundError(f"Template not found: {target}")
        return target.read_text(encoding="utf-8")

    def _substitute_variables(self, template: str, variables: dict[str, str]) -> str:
        result = template
        for key, value in variables.items():
            token = f"{self._VAR_PREFIX}{key}{self._VAR_SUFFIX}"
            result = result.replace(token, value)
        return result

    def _replace_single_include(self, match: re.Match[str]) -> str:
        rel_path = match.group(1).strip()
        base_dir = Path(self._templates_dir).resolve()
        target_path = (base_dir / rel_path).resolve()
        if not str(target_path).startswith(str(base_dir)):
            raise ValueError(f"Path traversal detected in include: {rel_path}")
        if not target_path.is_file():
            raise FileNotFoundError(f"Included file does not exist: {target_path}")
        return target_path.read_text(encoding="utf-8")
