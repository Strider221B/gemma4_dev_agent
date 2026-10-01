"""Git operations and unified diff parsing utilities."""

from __future__ import annotations

import subprocess


class GitUtils:
    """Utility class for git repository operations and unified diff parsing."""

    _DIFF_PREFIX: str = "diff --git"
    _HEADER_OLD: str = "--- "
    _HEADER_NEW: str = "+++ "
    _PREFIX_ADD: str = "+"
    _PREFIX_DEL: str = "-"
    _HEADER_DIFF3: str = "+++"
    _HEADER_DIFF3_DEL: str = "---"
    _GIT_CMD: str = "git"
    _EXIT_SUCCESS: int = 0
    _EXIT_FAILURE: int = 1

    def __init__(self) -> None:
        """Initialize GitUtils instance."""
        pass

    def parse_unified_diff(self, diff_text: str) -> list[dict[str, object]]:
        """Parse raw unified diff string into structured file entries."""
        if not diff_text.strip():
            return []
        raw_blocks = diff_text.split(self._DIFF_PREFIX)
        blocks = [f"{self._DIFF_PREFIX}{b}" for b in raw_blocks if b.strip()]
        if not blocks and diff_text.strip().startswith((self._HEADER_OLD, "@@")):
            blocks = [diff_text]
        return [self._parse_file_block(b) for b in blocks]

    def apply_patch(self, workspace_path: str, patch_text: str) -> bool:
        """Apply unified diff patch to workspace working tree."""
        if not patch_text.strip():
            return True
        code, _, _ = self._run_git_command(
            workspace_path, ["apply", "--whitespace=nowarn", "-"], input_data=patch_text
        )
        return code == self._EXIT_SUCCESS

    def list_changed_files(self, workspace_path: str) -> list[str]:
        """List files with uncommitted changes in workspace."""
        code, stdout, _ = self._run_git_command(
            workspace_path, ["diff", "--name-only"]
        )
        if code != self._EXIT_SUCCESS:
            return []
        return [line.strip() for line in stdout.splitlines() if line.strip()]

    def checkout_file(self, workspace_path: str, filepath: str) -> bool:
        """Restore file from HEAD in workspace."""
        code, _, _ = self._run_git_command(
            workspace_path, ["checkout", "HEAD", "--", filepath]
        )
        return code == self._EXIT_SUCCESS

    def _parse_file_block(self, block: str) -> dict[str, object]:
        """Extract path metadata and change counts from a single file diff block."""
        lines = block.splitlines()
        old_path, new_path = self._extract_paths(lines)
        added, deleted = self._count_line_changes(lines)
        return {
            "old_path": old_path,
            "new_path": new_path,
            "diff": block,
            "added_lines": added,
            "deleted_lines": deleted,
        }

    def _extract_paths(self, lines: list[str]) -> tuple[str, str]:
        """Extract source and target file paths from diff header lines."""
        old_path = ""
        new_path = ""
        for line in lines:
            if line.startswith(self._HEADER_OLD) and not old_path:
                old_path = line[len(self._HEADER_OLD):].strip()
            elif line.startswith(self._HEADER_NEW) and not new_path:
                new_path = line[len(self._HEADER_NEW):].strip()
        return old_path, new_path

    def _count_line_changes(self, lines: list[str]) -> tuple[int, int]:
        """Count added and removed lines within a diff block excluding header lines."""
        added = 0
        deleted = 0
        for line in lines:
            if line.startswith(self._PREFIX_ADD) and not line.startswith(self._HEADER_DIFF3):
                added += 1
            elif line.startswith(self._PREFIX_DEL) and not line.startswith(self._HEADER_DIFF3_DEL):
                deleted += 1
        return added, deleted

    def _run_git_command(
        self, workspace_path: str, args: list[str], input_data: str | None = None
    ) -> tuple[int, str, str]:
        """Execute git command via subprocess in target workspace."""
        try:
            result = subprocess.run(
                [self._GIT_CMD, *args],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                input=input_data,
                check=False,
            )
            return result.returncode, result.stdout, result.stderr
        except OSError:
            return self._EXIT_FAILURE, "", "Failed to execute git process"
