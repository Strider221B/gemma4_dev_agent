"""Local mock pipeline test runner validating pipeline wiring without GPU."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import ClassVar

from src.config.config_manager import ConfigManager
from src.config.deploy_config import DeployConfig
from src.data.dataset_builder import DatasetBuilder
from src.data.mock_data_factory import MockDataFactory
from src.data.task import Task
from src.deployment.constraint_validator import ConstraintValidator
from src.deployment.submission_packager import SubmissionPackager
from src.evaluation.cv_splitter import CVSplitter
from src.training.checkpoint_manager import CheckpointManager
from src.training.curriculum_scheduler import CurriculumScheduler
from src.utils.telemetry_logger import TelemetryLogger


class MockNotebook:
    """Orchestrates end-to-end mock execution of the post-training and submission pipeline."""

    _CONFIG_PATH: ClassVar[str] = "configs/sft_config.yaml"
    _TEMPLATES_DIR: ClassVar[str] = "kaggle_staging/submission_templates"
    _VERSION: ClassVar[str] = "v0.1.0-mock"
    _ADAPTER_DIR_NAME: ClassVar[str] = "sft_lora"
    _ADAPTER_CONFIG_NAME: ClassVar[str] = "adapter_config.json"
    _ADAPTER_WEIGHTS_NAME: ClassVar[str] = "adapter_model.safetensors"
    _ADAPTER_NAME: ClassVar[str] = "coder_lora"
    _SUBMISSION_ZIP_NAME: ClassVar[str] = "submission.zip"
    _TEMP_DIR_PREFIX: ClassVar[str] = "mock_pipeline_"
    _LOG_DIR_NAME: ClassVar[str] = "logs"
    _SUBMISSION_DIR_NAME: ClassVar[str] = "submission"
    _DEFAULT_LORA_R: ClassVar[int] = 16
    _DEFAULT_LORA_ALPHA: ClassVar[int] = 32
    _DUMMY_WEIGHT_SIZE: ClassVar[int] = 1024
    _CURRICULUM_EPOCH: ClassVar[int] = 1
    _CURRICULUM_TOTAL_EPOCHS: ClassVar[int] = 5
    _MOCK_REPO_PREFIX: ClassVar[str] = "mock/repo_"
    _MOCK_TASK_PREFIX: ClassVar[str] = "mock_task_"

    def __init__(self, work_dir: str | None = None) -> None:
        """Initialize MockNotebook with optional working directory."""
        self._work_dir: str = work_dir or tempfile.mkdtemp(prefix=self._TEMP_DIR_PREFIX)
        self._owns_work_dir: bool = work_dir is None

    def run(self) -> str:
        """Execute mock pipeline steps and return path to generated submission zip."""
        try:
            telemetry = self._init_telemetry()
            config = self._load_configuration()
            dataset = self._build_dataset(config)
            adapter_path = self._run_mock_training(dataset)
            self._verify_cv_splitting()
            zip_path = self._package_submission(adapter_path)
            telemetry.log_info(f"Mock pipeline completed successfully: {zip_path}")
            return zip_path
        finally:
            if self._owns_work_dir:
                self._cleanup()

    def _init_telemetry(self) -> TelemetryLogger:
        log_dir = os.path.join(self._work_dir, self._LOG_DIR_NAME)
        os.makedirs(log_dir, exist_ok=True)
        return TelemetryLogger(log_dir=log_dir, run_version=self._VERSION)

    def _load_configuration(self) -> ConfigManager:
        return ConfigManager.load(self._CONFIG_PATH)

    def _build_dataset(self, config: ConfigManager) -> object:
        builder = DatasetBuilder(mock_mode=True)
        dataset = builder.build()
        print(f"[1/5] Dataset built successfully with {len(dataset)} examples")
        return dataset

    def _run_mock_training(self, dataset: object) -> str:
        curriculum = CurriculumScheduler()
        filtered = curriculum.get_epoch_data(
            dataset, self._CURRICULUM_EPOCH, self._CURRICULUM_TOTAL_EPOCHS
        )
        print(f"[2/5] Curriculum scheduler filtered dataset to {len(filtered)} items")
        adapter_path = os.path.join(self._work_dir, self._ADAPTER_DIR_NAME)
        self._create_mock_checkpoint(adapter_path)
        mgr = CheckpointManager()
        if not mgr.validate_size(adapter_path):
            raise ValueError("Mock adapter size validation failed")
        print(f"[3/5] Mock adapter checkpoint verified at {adapter_path}")
        return adapter_path

    def _verify_cv_splitting(self) -> None:
        tasks = self._create_split_tasks()
        splitter = CVSplitter()
        splits = splitter.create_splits(tasks)
        print(f"[4/5] CV splitter verified with {len(splits)} folds")

    def _create_split_tasks(self) -> list[Task]:
        factory = MockDataFactory()
        base_task = factory.create_mock_task()
        tasks: list[Task] = [base_task]
        for idx in range(1, 3):
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

    def _package_submission(self, adapter_path: str) -> str:
        out_dir = os.path.join(self._work_dir, self._SUBMISSION_DIR_NAME)
        zip_path = os.path.join(self._work_dir, self._SUBMISSION_ZIP_NAME)
        deploy_cfg = DeployConfig(
            templates_dir=self._TEMPLATES_DIR,
            output_dir=out_dir,
            zip_path=zip_path,
            adapters=[{"name": self._ADAPTER_NAME, "checkpoint_path": adapter_path}],
        )
        packager = SubmissionPackager(validator=ConstraintValidator())
        res_zip = packager.package(deploy_cfg)
        size_kb = os.path.getsize(res_zip) / 1024.0
        print(f"[5/5] Submission packaged successfully: {res_zip} ({size_kb:.1f} KB)")
        return res_zip

    def _create_mock_checkpoint(self, dest_dir: str) -> None:
        os.makedirs(dest_dir, exist_ok=True)
        config_path = os.path.join(dest_dir, self._ADAPTER_CONFIG_NAME)
        adapter_cfg = {
            "r": self._DEFAULT_LORA_R,
            "lora_alpha": self._DEFAULT_LORA_ALPHA,
            "target_modules": ["q_proj", "v_proj"],
        }
        with open(config_path, "w", encoding="utf-8") as file_handle:
            json.dump(adapter_cfg, file_handle)
        weights_path = os.path.join(dest_dir, self._ADAPTER_WEIGHTS_NAME)
        with open(weights_path, "wb") as file_handle:
            file_handle.write(b"\x00" * self._DUMMY_WEIGHT_SIZE)

    def _cleanup(self) -> None:
        if os.path.exists(self._work_dir):
            shutil.rmtree(self._work_dir, ignore_errors=True)


if __name__ == "__main__":
    runner = MockNotebook()
    runner.run()
