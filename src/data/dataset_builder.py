"""Dataset builder module orchestrating the end-to-end data pipeline."""

from __future__ import annotations

from src.config.data_paths_config import DataPathsConfig
from src.data.chat_formatter import ChatFormatter
from src.data.constants import MAX_TRAJECTORY_TOKENS
from src.data.in_memory_dataset import InMemoryDataset
from src.data.ingestor import DataIngestor
from src.data.mock_data_factory import MockDataFactory
from src.data.patch_parser import PatchParser
from src.data.task import Task
from src.data.trajectory import Trajectory
from src.data.trajectory_augmentor import TrajectoryAugmentor
from src.data.trajectory_synthesiser import TrajectorySynthesiser
from src.data.trajectory_validator import TrajectoryValidator
from src.evaluation.cv_splitter import CVSplitter
from src.utils.token_counter import TokenCounter


class DatasetBuilder:
    """Orchestrates ingestion, synthesis, augmentation, validation, and splitting."""

    _DEFAULT_FOLD: int = 0
    _KEY_INSTANCE_ID: str = "instance_id"
    _KEY_REPO: str = "repo"
    _KEY_COMPLEXITY: str = "complexity"
    _KEY_MESSAGES: str = "messages"
    _KEY_FORMATTED_TEXT: str = "formatted_text"
    _KEY_TOKEN_COUNT: str = "token_count"
    _KEY_NUM_TOOL_CALLS: str = "num_tool_calls"
    _KEY_NUM_FILES_CHANGED: str = "num_files_changed"
    _KEY_FOLD: str = "fold"
    _EMPTY_STR: str = ""
    _MOCK_REPO_PREFIX: str = "mock/repo_"
    _MOCK_TASK_PREFIX: str = "mock_utils_"

    def __init__(
        self,
        ingestor: DataIngestor | None = None,
        synthesiser: TrajectorySynthesiser | None = None,
        augmentor: TrajectoryAugmentor | None = None,
        formatter: ChatFormatter | None = None,
        validator: TrajectoryValidator | None = None,
        splitter: CVSplitter | None = None,
        mock_mode: bool = False,
    ) -> None:
        """Initialize DatasetBuilder with injected or default pipeline components."""
        paths = DataPathsConfig(tasks_path="", graphs_dir="", embeddings_dir="")
        self._ingestor: DataIngestor = ingestor or DataIngestor(paths)
        token_counter = TokenCounter()
        self._synthesiser: TrajectorySynthesiser = synthesiser or (
            TrajectorySynthesiser(
                self._ingestor, PatchParser(), token_counter, mock_mode=mock_mode
            )
        )
        self._augmentor: TrajectoryAugmentor = (
            augmentor if augmentor is not None else TrajectoryAugmentor()
        )
        self._formatter: ChatFormatter = (
            formatter if formatter is not None else ChatFormatter(token_counter)
        )
        self._validator: TrajectoryValidator = (
            validator if validator is not None else TrajectoryValidator(token_counter)
        )
        self._splitter: CVSplitter = (
            splitter if splitter is not None else CVSplitter()
        )
        self._mock_mode: bool = mock_mode

    def build(self) -> object:
        """Execute full pipeline and return assembled HuggingFace Dataset."""
        if self._mock_mode:
            return self.build_mock()
        tasks = self._load_tasks()
        return self._process_tasks(tasks)

    def build_mock(self) -> object:
        """Execute mock pipeline using MockDataFactory tasks."""
        tasks = self._generate_mock_tasks()
        return self._process_tasks(tasks)

    def _load_tasks(self) -> list[Task]:
        """Delegate task loading to the injected data ingestor."""
        return self._ingestor.load_tasks()

    def _synthesise_trajectories(self, tasks: list[Task]) -> list[Trajectory]:
        """Synthesise agent interaction trajectories for given tasks."""
        return self._synthesiser.synthesise_all(tasks)

    def _augment_trajectories(
        self, trajectories: list[Trajectory]
    ) -> list[Trajectory]:
        """Produce augmented variants for each input trajectory."""
        augmented: list[Trajectory] = []
        for trajectory in trajectories:
            augmented.extend(self._augmentor.augment(trajectory))
        return augmented

    def _filter_by_token_budget(
        self, trajectories: list[Trajectory]
    ) -> list[Trajectory]:
        """Filter out trajectories exceeding maximum allowed token budget."""
        return [
            t for t in trajectories if t.token_count <= MAX_TRAJECTORY_TOKENS
        ]

    def _validate_trajectories(
        self, trajectories: list[Trajectory]
    ) -> list[Trajectory]:
        """Filter out trajectories that fail validation checks."""
        valid: list[Trajectory] = []
        for trajectory in trajectories:
            result = self._validator.validate(trajectory)
            if result.valid:
                valid.append(trajectory)
        return valid

    def _format_trajectories(
        self, trajectories: list[Trajectory]
    ) -> list[dict[str, object]]:
        """Format validated trajectories into training dataset records."""
        records: list[dict[str, object]] = []
        for trajectory in trajectories:
            records.append(self._build_record(trajectory))
        return records

    def _build_record(self, trajectory: Trajectory) -> dict[str, object]:
        """Construct a single row dictionary from trajectory metadata."""
        return {
            self._KEY_INSTANCE_ID: trajectory.instance_id,
            self._KEY_REPO: trajectory.repo,
            self._KEY_COMPLEXITY: trajectory.complexity.value,
            self._KEY_MESSAGES: self._formatter.format_messages(trajectory.turns),
            self._KEY_FORMATTED_TEXT: self._formatter.format_trajectory(trajectory),
            self._KEY_TOKEN_COUNT: trajectory.token_count,
            self._KEY_NUM_TOOL_CALLS: trajectory.num_tool_calls,
            self._KEY_NUM_FILES_CHANGED: trajectory.num_files_changed,
            self._KEY_FOLD: self._DEFAULT_FOLD,
        }

    def _assign_folds(
        self, records: list[dict[str, object]], tasks: list[Task]
    ) -> list[dict[str, object]]:
        """Assign repository-grouped cross-validation folds to records."""
        try:
            splits = self._splitter.create_splits(tasks)
        except Exception:
            splits = []
        fold_map: dict[str, int] = {}
        for fold in splits:
            for val_id in fold.val_ids:
                fold_map[val_id] = fold.fold_idx
        for record in records:
            inst_id = str(record.get(self._KEY_INSTANCE_ID, self._EMPTY_STR))
            record[self._KEY_FOLD] = fold_map.get(inst_id, self._DEFAULT_FOLD)
        return records

    def _build_hf_dataset(self, records: list[dict[str, object]]) -> object:
        """Create HuggingFace Dataset or fallback InMemoryDataset."""
        try:
            from datasets import Dataset

            return Dataset.from_list(records)
        except (ImportError, Exception):
            return InMemoryDataset(records)

    def _generate_mock_tasks(self) -> list[Task]:
        """Generate 4 mock tasks across distinct repositories for fold splitting."""
        factory = MockDataFactory()
        base_task = factory.create_mock_task()
        tasks: list[Task] = [base_task]
        for idx in range(1, 4):
            tasks.append(
                Task(
                    instance_id=f"{self._MOCK_TASK_PREFIX}{idx:03d}",
                    repo=f"{self._MOCK_REPO_PREFIX}{idx}",
                    base_commit=base_task.base_commit,
                    problem_statement=base_task.problem_statement,
                    hints_text=base_task.hints_text,
                    patch=base_task.patch,
                    test_patch=base_task.test_patch,
                    created_at=base_task.created_at,
                )
            )
        return tasks

    def _process_tasks(self, tasks: list[Task]) -> object:
        """Execute end-to-end dataset transformation on given tasks."""
        synthesised = self._synthesise_trajectories(tasks)
        augmented = self._augment_trajectories(synthesised)
        all_trajectories = synthesised + augmented
        budget_checked = self._filter_by_token_budget(all_trajectories)
        validated = self._validate_trajectories(budget_checked)
        records = self._format_trajectories(validated)
        assigned_records = self._assign_folds(records, tasks)
        return self._build_hf_dataset(assigned_records)
