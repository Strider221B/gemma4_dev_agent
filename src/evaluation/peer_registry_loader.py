"""Loader and serializer for peer solution registry YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.evaluation.peer_solution import PeerSolution


class PeerRegistryLoader:
    """Manages loading, updating, and saving the peer solution catalog in YAML format."""

    _KEY_SOLUTIONS: str = "solutions"
    _KEY_NAME: str = "name"
    _KEY_NOTEBOOK_NAME: str = "notebook_name"
    _KEY_KAGGLE_URL: str = "kaggle_url"
    _KEY_AUTHOR: str = "author"
    _KEY_LB_SCORE: str = "lb_score"
    _KEY_CAPTURE_DATE: str = "capture_date"
    _KEY_LOCAL_PATH: str = "local_path"
    _KEY_STATUS: str = "status"
    _KEY_ANALYSIS_STATUS: str = "analysis_status"
    _KEY_PROMPT_STRATEGY: str = "prompt_strategy"
    _KEY_AGENT_ARCH: str = "agent_architecture"
    _KEY_ADAPTER_DETAILS: str = "adapter_details"
    _KEY_KEY_TECHNIQUES: str = "key_techniques"
    _KEY_CV_SCORE: str = "cv_score"
    _KEY_CV_LB_GAP: str = "cv_lb_gap"
    _KEY_LEAK_DETECTED: str = "leak_detected"
    _STATUS_PENDING: str = "pending"
    _ENCODING: str = "utf-8"
    _DEFAULT_SCORE: float = 0.0

    def __init__(self, registry_path: str) -> None:
        """Initialize registry loader with target file path."""
        self._registry_path: str = registry_path

    def load_registry(self) -> list[PeerSolution]:
        """Parse peer registry YAML into a list of PeerSolution objects."""
        path = Path(self._registry_path)
        if not path.exists():
            return []
        content = path.read_text(encoding=self._ENCODING)
        data: Any = yaml.safe_load(content)
        if not isinstance(data, dict):
            return []
        raw_solutions = data.get(self._KEY_SOLUTIONS, [])
        if not isinstance(raw_solutions, list):
            return []
        return [
            self._parse_solution(entry)
            for entry in raw_solutions
            if isinstance(entry, dict)
        ]

    def save_registry(self, solutions: list[PeerSolution]) -> None:
        """Serialize a list of PeerSolution instances to the registry YAML file."""
        serialized = [self._serialize_solution(s) for s in solutions]
        payload = {self._KEY_SOLUTIONS: serialized}
        path = Path(self._registry_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dumped = yaml.dump(payload, sort_keys=False)
        path.write_text(dumped, encoding=self._ENCODING)

    def add_solution(self, solution: PeerSolution) -> None:
        """Append a new solution to the registry and write to disk."""
        solutions = self.load_registry()
        solutions.append(solution)
        self.save_registry(solutions)

    def update_status(
        self,
        notebook_name: str,
        status: str,
        cv_score: float | None = None,
        decision: str | None = None,
    ) -> None:
        """Update evaluation status and optional metrics for a registered solution."""
        solutions = self.load_registry()
        updated = False
        for s in solutions:
            if s.notebook_name == notebook_name:
                s.analysis_status = status
                if cv_score is not None:
                    s.cv_score = cv_score
                updated = True
        if updated:
            self.save_registry(solutions)

    def _parse_solution(self, entry: dict[str, object]) -> PeerSolution:
        """Parse dictionary entry from YAML into PeerSolution."""
        raw_lb = entry.get(self._KEY_LB_SCORE)
        lb_score = float(raw_lb) if raw_lb is not None else self._DEFAULT_SCORE  # type: ignore[arg-type]
        raw_cv = entry.get(self._KEY_CV_SCORE)
        cv_score = float(raw_cv) if raw_cv is not None else None  # type: ignore[arg-type]
        raw_gap = entry.get(self._KEY_CV_LB_GAP)
        cv_lb_gap = float(raw_gap) if raw_gap is not None else None  # type: ignore[arg-type]
        raw_leak = entry.get(self._KEY_LEAK_DETECTED)
        leak_detected = bool(raw_leak) if raw_leak is not None else None
        techs = entry.get(self._KEY_KEY_TECHNIQUES)
        key_techs = [str(t) for t in techs] if isinstance(techs, list) else []
        name = str(entry.get(self._KEY_NOTEBOOK_NAME) or entry.get(self._KEY_NAME) or "")
        status = str(
            entry.get(self._KEY_ANALYSIS_STATUS)
            or entry.get(self._KEY_STATUS)
            or self._STATUS_PENDING
        )
        return PeerSolution(
            notebook_name=name,
            kaggle_url=str(entry.get(self._KEY_KAGGLE_URL) or ""),
            author=str(entry.get(self._KEY_AUTHOR) or ""),
            lb_score=lb_score,
            capture_date=str(entry.get(self._KEY_CAPTURE_DATE) or ""),
            local_path=str(entry.get(self._KEY_LOCAL_PATH) or ""),
            analysis_status=status,
            prompt_strategy=(
                str(entry[self._KEY_PROMPT_STRATEGY])
                if entry.get(self._KEY_PROMPT_STRATEGY)
                else None
            ),
            agent_architecture=(
                str(entry[self._KEY_AGENT_ARCH])
                if entry.get(self._KEY_AGENT_ARCH)
                else None
            ),
            adapter_details=(
                str(entry[self._KEY_ADAPTER_DETAILS])
                if entry.get(self._KEY_ADAPTER_DETAILS)
                else None
            ),
            key_techniques=key_techs,
            cv_score=cv_score,
            cv_lb_gap=cv_lb_gap,
            leak_detected=leak_detected,
        )

    def _serialize_solution(self, solution: PeerSolution) -> dict[str, object]:
        """Serialize a PeerSolution dataclass into a YAML-compatible dict."""
        return {
            self._KEY_NAME: solution.notebook_name,
            self._KEY_NOTEBOOK_NAME: solution.notebook_name,
            self._KEY_AUTHOR: solution.author,
            self._KEY_KAGGLE_URL: solution.kaggle_url,
            self._KEY_LB_SCORE: solution.lb_score,
            self._KEY_CAPTURE_DATE: solution.capture_date,
            self._KEY_LOCAL_PATH: solution.local_path,
            self._KEY_STATUS: solution.analysis_status,
            self._KEY_ANALYSIS_STATUS: solution.analysis_status,
            self._KEY_PROMPT_STRATEGY: solution.prompt_strategy,
            self._KEY_AGENT_ARCH: solution.agent_architecture,
            self._KEY_ADAPTER_DETAILS: solution.adapter_details,
            self._KEY_KEY_TECHNIQUES: solution.key_techniques,
            self._KEY_CV_SCORE: solution.cv_score,
            self._KEY_CV_LB_GAP: solution.cv_lb_gap,
            self._KEY_LEAK_DETECTED: solution.leak_detected,
        }
