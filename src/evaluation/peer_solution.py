"""Data model representing a peer notebook solution under competitive analysis."""

from dataclasses import dataclass, field


@dataclass
class PeerSolution:
    """Represents metadata and evaluation state of a peer competitor solution."""

    notebook_name: str
    kaggle_url: str
    author: str
    lb_score: float
    capture_date: str
    local_path: str
    analysis_status: str = "pending"
    prompt_strategy: str | None = None
    agent_architecture: str | None = None
    adapter_details: str | None = None
    key_techniques: list[str] = field(default_factory=list)
    cv_score: float | None = None
    cv_lb_gap: float | None = None
    leak_detected: bool | None = None
