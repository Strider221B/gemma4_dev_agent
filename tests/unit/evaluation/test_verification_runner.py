"""Unit tests for VerificationRunner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from src.data.task import Task
from src.evaluation.mock_sandbox import MockSandbox
from src.evaluation.verification_runner import VerificationRunner
from src.utils.telemetry_logger import TelemetryLogger


class TestVerificationRunner:
    """Test suite for VerificationRunner patch testing and JUnit validation."""

    _MOCK_COMMIT: str = "1234567890abcdef"
    _MOCK_CREATED: str = "2026-01-01T00:00:00Z"
    _XML_PASS: str = (
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0"/></testsuites>'
    )
    _XML_FAIL: str = (
        '<testsuites><testsuite tests="2" failures="1" errors="0" skipped="0"/></testsuites>'
    )
    _XML_MALFORMED: str = "<not closed"

    def test_verify_with_valid_patch_returns_resolved(self) -> None:
        """Verify patch application and passing tests return resolved=True."""
        sandbox = MagicMock(spec=MockSandbox)
        sandbox.apply_patch.return_value = True
        sandbox.run_command.return_value = {"exit_code": 0, "stdout": "", "stderr": ""}
        telemetry = MagicMock(spec=TelemetryLogger)
        runner = VerificationRunner(sandbox=sandbox, telemetry=telemetry)
        runner._get_junit_path = MagicMock(return_value=None)
        runner._fallback_counts = MagicMock(
            return_value={"tests": 2, "passed": 2, "failures": 0, "errors": 0}
        )
        task = self._create_task()
        result = runner.verify("diff --git a/file.py", task)
        assert result.resolved is True
        assert result.error is None
        assert result.passed_tests == 2
        sandbox.setup_workspace.assert_called_once_with(task)
        sandbox.cleanup.assert_called_once()

    def test_verify_with_invalid_patch_returns_not_resolved(self) -> None:
        """Verify failed patch application returns resolved=False."""
        sandbox = MagicMock(spec=MockSandbox)
        sandbox.apply_patch.return_value = False
        telemetry = MagicMock(spec=TelemetryLogger)
        runner = VerificationRunner(sandbox=sandbox, telemetry=telemetry)
        task = self._create_task()
        result = runner.verify("invalid diff", task)
        assert result.resolved is False
        assert result.error == "Patch application failed"
        sandbox.cleanup.assert_called_once()

    def test_verify_resets_protected_files(self) -> None:
        """Verify runner resets protected test files after applying patch."""
        sandbox = MagicMock(spec=MockSandbox)
        sandbox.apply_patch.return_value = True
        sandbox.run_command.return_value = {"exit_code": 0, "stdout": "", "stderr": ""}
        telemetry = MagicMock(spec=TelemetryLogger)
        runner = VerificationRunner(sandbox=sandbox, telemetry=telemetry)
        task = self._create_task()
        runner.verify("diff --git", task)
        sandbox.reset_protected_files.assert_called_once_with(task)

    def test_verify_batch_evaluates_all(self) -> None:
        """Verify verify_batch executes verification for all patch/task pairs."""
        sandbox = MagicMock(spec=MockSandbox)
        sandbox.apply_patch.return_value = True
        sandbox.run_command.return_value = {"exit_code": 0, "stdout": "", "stderr": ""}
        telemetry = MagicMock(spec=TelemetryLogger)
        runner = VerificationRunner(sandbox=sandbox, telemetry=telemetry)
        tasks = [self._create_task("t1"), self._create_task("t2")]
        patches = ["patch1", "patch2"]
        results = runner.verify_batch(patches, tasks)
        assert len(results) == 2

    def test_validate_junit_xml_parsing(self) -> None:
        """Verify _validate_junit_xml parses pass, failure, and malformed XML."""
        sandbox = MagicMock(spec=MockSandbox)
        telemetry = MagicMock(spec=TelemetryLogger)
        runner = VerificationRunner(sandbox=sandbox, telemetry=telemetry)
        counts_pass = runner._validate_junit_xml(self._XML_PASS)
        assert counts_pass["tests"] == 2
        assert counts_pass["passed"] == 2
        assert counts_pass["failures"] == 0

        counts_fail = runner._validate_junit_xml(self._XML_FAIL)
        assert counts_fail["tests"] == 2
        assert counts_fail["passed"] == 1
        assert counts_fail["failures"] == 1

        counts_bad = runner._validate_junit_xml(self._XML_MALFORMED)
        assert counts_bad["tests"] == 0

    def test_verify_test_patch_failure_returns_not_resolved(self) -> None:
        """Verify failure to apply test patch returns error result."""
        sandbox = MagicMock(spec=MockSandbox)
        sandbox.apply_patch.return_value = True
        sandbox._apply_test_patch.return_value = False
        telemetry = MagicMock(spec=TelemetryLogger)
        runner = VerificationRunner(sandbox=sandbox, telemetry=telemetry)
        task = self._create_task(test_patch="diff --git test")
        res = runner.verify("patch", task)
        assert res.resolved is False
        assert res.error == "Test patch application failed"

    def test_verify_test_suite_failures_returns_not_resolved(self) -> None:
        """Verify non-zero exit code or test failures mark resolved=False."""
        sandbox = MagicMock(spec=MockSandbox)
        sandbox.apply_patch.return_value = True
        sandbox.run_command.return_value = {"exit_code": 1, "stdout": "", "stderr": "FAILED"}
        telemetry = MagicMock(spec=TelemetryLogger)
        runner = VerificationRunner(sandbox=sandbox, telemetry=telemetry)
        runner._fallback_counts = MagicMock(
            return_value={"tests": 1, "passed": 0, "failures": 1, "errors": 0}
        )
        task = self._create_task()
        res = runner.verify("patch", task)
        assert res.resolved is False
        assert res.failed_tests == 1
        assert res.error is not None

    def test_extract_counts_from_file_and_get_junit_path(self, tmp_path: Path) -> None:
        """Verify reading junit report from file and getting path."""
        xml_file = tmp_path / "junit.xml"
        xml_file.write_text(self._XML_PASS, encoding="utf-8")
        sandbox = MagicMock(spec=MockSandbox)
        sandbox.get_workspace_path.return_value = str(tmp_path)
        telemetry = MagicMock(spec=TelemetryLogger)
        runner = VerificationRunner(sandbox=sandbox, telemetry=telemetry)
        p = runner._get_junit_path(sandbox)
        assert p == xml_file
        counts = runner._extract_counts_from_file(xml_file)
        assert counts["passed"] == 2

    def test_to_int_cases(self) -> None:
        """Verify _to_int conversion across int, float, str, and invalid types."""
        sandbox = MagicMock(spec=MockSandbox)
        telemetry = MagicMock(spec=TelemetryLogger)
        runner = VerificationRunner(sandbox=sandbox, telemetry=telemetry)
        assert runner._to_int(5) == 5
        assert runner._to_int("10") == 10
        assert runner._to_int(3.8) == 3
        assert runner._to_int("bad", default=42) == 42
        assert runner._to_int(None, default=7) == 7

    def test_tally_test_suites_single_root_suite(self) -> None:
        """Verify _validate_junit_xml with root tag testsuite."""
        sandbox = MagicMock(spec=MockSandbox)
        telemetry = MagicMock(spec=TelemetryLogger)
        runner = VerificationRunner(sandbox=sandbox, telemetry=telemetry)
        xml = '<testsuite tests="3" failures="0" errors="0" skipped="1"/>'
        counts = runner._validate_junit_xml(xml)
        assert counts["tests"] == 3
        assert counts["passed"] == 2

    def _create_task(
        self, instance_id: str = "test_001", test_patch: str = ""
    ) -> Task:
        """Helper to create dummy Task."""
        return Task(
            instance_id=instance_id,
            repo="tiangolo/fastapi",
            base_commit=self._MOCK_COMMIT,
            problem_statement="Description",
            hints_text="Hints",
            patch="diff patch",
            test_patch=test_patch,
            created_at=self._MOCK_CREATED,
        )
