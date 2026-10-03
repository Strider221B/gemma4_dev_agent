"""Generator for constructing preference pairs used in Direct Preference Optimization (DPO)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.task import Task
    from src.data.trajectory import Trajectory
    from src.training.reward_model import RewardModel
    from src.utils.telemetry_logger import TelemetryLogger


class PreferencePairGenerator:
    """Generates (chosen, rejected) preference trajectory pairs ranked by reward score."""

    _MIN_SCORE_DIFFERENCE: float = 0.3
    _KEY_INSTANCE_ID: str = "instance_id"
    _KEY_PROMPT: str = "prompt"
    _KEY_CHOSEN: str = "chosen"
    _KEY_REJECTED: str = "rejected"
    _KEY_CHOSEN_SCORE: str = "chosen_score"
    _KEY_REJECTED_SCORE: str = "rejected_score"
    _PROMPT_TEMPLATE: str = "<|turn>user\nProblem: {problem}\n<turn|>\n"
    _THOUGHT_WRAPPER: str = "<|channel>thought\n{content}\n<channel|>"
    _JOIN_NEWLINE: str = "\n"

    def __init__(
        self, reward_model: RewardModel, telemetry: TelemetryLogger
    ) -> None:
        """Initialize PreferencePairGenerator with injected reward model and telemetry."""
        self._reward_model: RewardModel = reward_model
        self._telemetry: TelemetryLogger = telemetry

    def generate_pairs(
        self, tasks: list[Task], model: object, tokenizer: object, n_per_task: int = 4
    ) -> list[dict[str, object]]:
        """Generate pairwise preferences where chosen score strictly exceeds rejected score."""
        self._telemetry.log_info(f"Generating preference pairs for {len(tasks)} tasks")
        pairs: list[dict[str, object]] = []
        for task in tasks:
            completions = self._generate_completions(
                model, tokenizer, task, n_per_task
            )
            scored = self._score_and_rank(completions, task)
            for i in range(len(scored)):
                for j in range(i + 1, len(scored)):
                    chosen, chosen_score = scored[i]
                    rejected, rejected_score = scored[j]
                    if (chosen_score - rejected_score) > self._MIN_SCORE_DIFFERENCE:
                        pair = self._create_pair(
                            chosen, rejected, task, chosen_score, rejected_score
                        )
                        pairs.append(pair)
        return pairs

    def _build_prompt(self, task: Task) -> str:
        """Construct user turn prompt string from task problem statement."""
        return self._PROMPT_TEMPLATE.format(problem=task.problem_statement)

    def _create_pair(
        self,
        chosen: Trajectory,
        rejected: Trajectory,
        task: Task,
        chosen_score: float = 0.0,
        rejected_score: float = 0.0,
    ) -> dict[str, object]:
        """Construct preference dictionary payload for DPO training."""
        return {
            self._KEY_INSTANCE_ID: task.instance_id,
            self._KEY_PROMPT: self._build_prompt(task),
            self._KEY_CHOSEN: self._format_trajectory(chosen),
            self._KEY_REJECTED: self._format_trajectory(rejected),
            self._KEY_CHOSEN_SCORE: chosen_score,
            self._KEY_REJECTED_SCORE: rejected_score,
        }

    def _format_trajectory(self, trajectory: Trajectory) -> str:
        """Format trajectory into string representation for preference pair."""
        if hasattr(trajectory, "formatted_text"):
            return str(getattr(trajectory, "formatted_text"))
        parts: list[str] = []
        for turn in trajectory.turns:
            if turn.thought:
                parts.append(self._THOUGHT_WRAPPER.format(content=turn.thought))
            if turn.text:
                parts.append(turn.text)
            if turn.tool_result:
                parts.append(turn.tool_result)
        if not parts and trajectory.patch:
            parts.append(trajectory.patch)
        return self._JOIN_NEWLINE.join(parts)

    def _generate_completions(
        self, model: object, tokenizer: object, task: Task, n: int
    ) -> list[Trajectory]:
        """Generate completion trajectories for a given task using rollout generator."""
        from src.training.rollout_generator import RolloutGenerator

        generator = RolloutGenerator(self._telemetry)
        return generator.generate_rollouts(model, tokenizer, task, n)

    def _score_and_rank(
        self, completions: list[Trajectory], task: Task
    ) -> list[tuple[Trajectory, float]]:
        """Score each completion using reward model and sort descending by score."""
        scored = [
            (comp, self._reward_model.compute_reward(comp, task))
            for comp in completions
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored
