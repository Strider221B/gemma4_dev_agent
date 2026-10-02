"""Verification runner replicating Phase 2 evaluation container workflow."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src.data.task import Task
from src.evaluation.mock_sandbox import MockSandbox
from src.evaluation.verification_result import VerificationResult
from src.utils.telemetry_logger import TelemetryLogger


class VerificationRunner:
    """Executes patch application, anti-tampering resets, and test validation."""

    _PYTEST_COMMAND: str = "pytest --junitxml=junit.xml -q"
    _JUNIT_FILENAME: str = "junit.xml"
    _TEST_TIMEOUT: int = 120
    _LOG_PREFIX: str = "Verifying solution patch for task: "
    _ERR_PATCH_FAILED: str = "Patch application failed"
    _ERR_TEST_PATCH_FAILED: str = "Test patch application failed"
    _ERR_TESTS_FAILED: str = "Test suite failed or reported test failures"
    _KEY_EXIT_CODE: str = "exit_code"
    _KEY_STDOUT: str = "stdout"
    _KEY_PASSED_TESTS: str = "passed_tests"
    _KEY_FAILED_TESTS: str = "failed_tests"
    _KEY_TOTAL_TESTS: str = "total_tests"
    _KEY_ERRORS: str = "errors"
    _KEY_OUTPUT: str = "output"
    _KEY_TESTS: str = "tests"
    _KEY_PASSED: str = "passed"
    _KEY_FAILURES: str = "failures"
    _ATTR_TESTS: str = "tests"
    _ATTR_FAILURES: str = "failures"
    _ATTR_ERRORS: str = "errors"
    _ATTR_SKIPPED: str = "skipped"

    def __init__(self, sandbox: MockSandbox, telemetry: TelemetryLogger) -> None:
        """Initialize runner with sandbox and telemetry logger."""
        self._sandbox: MockSandbox = sandbox
        self._telemetry: TelemetryLogger = telemetry

    def verify(self, patch: str, task: Task) -> VerificationResult:
        """Apply patch and run test verification in an isolated sandbox."""
        self._telemetry.log_info(f"{self._LOG_PREFIX}{task.instance_id}")
        self._sandbox.setup_workspace(task)
        try:
            return self._execute_verification_workflow(patch, task)
        finally:
            self._sandbox.cleanup()

    def verify_batch(
        self, patches: list[str], tasks: list[Task]
    ) -> list[VerificationResult]:
        """Verify multiple patches against corresponding tasks."""
        return [
            self.verify(patch, task) for patch, task in zip(patches, tasks, strict=False)
        ]

    def _apply_patch(self, sandbox: MockSandbox, patch: str) -> bool:
        """Apply candidate patch to target sandbox."""
        return sandbox.apply_patch(patch)

    def _reset_protected_files(self, sandbox: MockSandbox, task: Task) -> None:
        """Replicate harness anti-tampering by reverting protected test modifications."""
        sandbox.reset_protected_files(task)

    def _run_pytest(self, sandbox: MockSandbox, task: Task) -> dict[str, object]:
        """Run pytest suite in sandbox and collect execution status and test counts."""
        cmd_res = sandbox.run_command(self._PYTEST_COMMAND, timeout=self._TEST_TIMEOUT)
        exit_code = self._to_int(cmd_res.get(self._KEY_EXIT_CODE), default=1)
        xml_path = self._get_junit_path(sandbox)
        counts = (
            self._extract_counts_from_file(xml_path)
            if xml_path and xml_path.exists()
            else self._fallback_counts(exit_code)
        )
        return {
            self._KEY_EXIT_CODE: exit_code,
            self._KEY_PASSED_TESTS: counts.get(self._KEY_PASSED, 0),
            self._KEY_FAILED_TESTS: counts.get(self._KEY_FAILURES, 0),
            self._KEY_TOTAL_TESTS: counts.get(self._KEY_TESTS, 0),
            self._KEY_ERRORS: counts.get(self._KEY_ERRORS, 0),
            self._KEY_OUTPUT: str(cmd_res.get(self._KEY_STDOUT, "")),
        }

    def _validate_junit_xml(self, xml_content: str) -> dict[str, int]:
        """Parse JUnit XML string and extract test outcome counts."""
        empty_counts = {
            self._KEY_TESTS: 0,
            self._KEY_PASSED: 0,
            self._KEY_FAILURES: 0,
            self._KEY_ERRORS: 0,
        }
        if not xml_content.strip():
            return empty_counts
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return empty_counts
        return self._tally_test_suites(root)

    def _execute_verification_workflow(self, patch: str, task: Task) -> VerificationResult:
        """Execute sequence of patch application, anti-tampering, and test run."""
        if not self._apply_patch(self._sandbox, patch):
            return VerificationResult(resolved=False, error=self._ERR_PATCH_FAILED)
        self._reset_protected_files(self._sandbox, task)
        if task.test_patch.strip():
            applied_test = self._sandbox._apply_test_patch(task.test_patch)
            if not applied_test:
                return VerificationResult(resolved=False, error=self._ERR_TEST_PATCH_FAILED)
        res = self._run_pytest(self._sandbox, task)
        return self._build_result_from_counts(res)

    def _build_result_from_counts(self, res: dict[str, object]) -> VerificationResult:
        """Construct VerificationResult from test run outcome metrics."""
        exit_code = self._to_int(res.get(self._KEY_EXIT_CODE), default=1)
        passed = self._to_int(res.get(self._KEY_PASSED_TESTS), default=0)
        failed = self._to_int(res.get(self._KEY_FAILED_TESTS), default=0)
        errors = self._to_int(res.get(self._KEY_ERRORS), default=0)
        total = self._to_int(res.get(self._KEY_TOTAL_TESTS), default=0)
        resolved = (exit_code == 0 and passed > 0 and failed == 0 and errors == 0)
        err = None if resolved else self._ERR_TESTS_FAILED
        return VerificationResult(
            resolved=resolved,
            error=err,
            passed_tests=passed,
            failed_tests=failed,
            total_tests=total,
        )

    def _to_int(self, value: object, default: int = 0) -> int:
        """Safely convert object to integer with fallback default value."""
        if isinstance(value, int):
            return value
        if isinstance(value, (str, float)):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    def _get_junit_path(self, sandbox: MockSandbox) -> Path | None:
        """Return path to generated junit xml report file if sandbox is active."""
        ws_path = sandbox.get_workspace_path()
        if ws_path:
            return Path(ws_path) / self._JUNIT_FILENAME
        return None

    def _extract_counts_from_file(self, xml_path: Path) -> dict[str, int]:
        """Read JUnit report file and return parsed test counters."""
        content = xml_path.read_text(encoding="utf-8")
        return self._validate_junit_xml(content)

    def _fallback_counts(self, exit_code: int) -> dict[str, int]:
        """Provide fallback test counts based solely on execution exit code."""
        if exit_code == 0:
            return {
                self._KEY_TESTS: 1,
                self._KEY_PASSED: 1,
                self._KEY_FAILURES: 0,
                self._KEY_ERRORS: 0,
            }
        return {
            self._KEY_TESTS: 1,
            self._KEY_PASSED: 0,
            self._KEY_FAILURES: 1,
            self._KEY_ERRORS: 0,
        }

    def _tally_test_suites(self, root: ET.Element) -> dict[str, int]:
        """Aggregate total, failure, error, and passed counts across XML suites."""
        suites = root.findall(".//testsuite")
        if not suites and root.tag == "testsuite":
            suites = [root]
        total_tests = sum(int(s.attrib.get(self._ATTR_TESTS, 0)) for s in suites)
        failures = sum(int(s.attrib.get(self._ATTR_FAILURES, 0)) for s in suites)
        errors = sum(int(s.attrib.get(self._ATTR_ERRORS, 0)) for s in suites)
        skipped = sum(int(s.attrib.get(self._ATTR_SKIPPED, 0)) for s in suites)
        passed = max(0, total_tests - failures - errors - skipped)
        return {
            self._KEY_TESTS: total_tests,
            self._KEY_PASSED: passed,
            self._KEY_FAILURES: failures,
            self._KEY_ERRORS: errors,
        }
