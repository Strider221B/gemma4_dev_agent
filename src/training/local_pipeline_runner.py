"""Local pipeline runner for end-to-end execution of training and packaging."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import ClassVar

from src.config.deploy_config import DeployConfig
from src.config.lora_config import LoRAConfig
from src.config.model_config import ModelConfig
from src.config.sft_config import SFTConfig
from src.config.training_config import TrainingConfig
from src.data.dataset_builder import DatasetBuilder
from src.deployment.constraint_validator import ConstraintValidator
from src.deployment.submission_packager import SubmissionPackager
from src.training.checkpoint_manager import CheckpointManager
from src.training.curriculum_scheduler import CurriculumScheduler
from src.training.sft_trainer import SFTTrainerPipeline
from src.training.surrogate_model_factory import SurrogateModelFactory
from src.utils.environment_detector import EnvironmentDetector
from src.utils.telemetry_logger import TelemetryLogger


class LocalPipelineRunner:
    """Orchestrates local training, checkpointing, and packaging workflows."""

    _MODE_SMOKE: ClassVar[str] = "smoke"
    _MODE_MOCK: ClassVar[str] = "mock"
    _MODE_FULL: ClassVar[str] = "full"
    _RUN_VERSION: ClassVar[str] = "v0.1.0-local"
    _ADAPTER_NAME: ClassVar[str] = "coder_lora"
    _DEFAULT_CONFIG_PATH: ClassVar[str] = "configs/sft_config.yaml"
    _TEMPLATES_DIR: ClassVar[str] = "kaggle_staging/submission_templates"
    _SUBDIR_SURROGATE: ClassVar[str] = "surrogate"
    _SUBDIR_CHECKPOINTS: ClassVar[str] = "checkpoints"
    _SUBDIR_LOGS: ClassVar[str] = "logs"
    _SUBDIR_SUBMISSION: ClassVar[str] = "submission"
    _FILE_LOG: ClassVar[str] = "run_sft.log"
    _FILE_SUBMISSION_ZIP: ClassVar[str] = "submission.zip"
    _FILE_ADAPTER_CONFIG: ClassVar[str] = "adapter_config.json"
    _FILE_ADAPTER_WEIGHTS: ClassVar[str] = "adapter_model.safetensors"
    _KEY_STATUS: ClassVar[str] = "status"
    _KEY_STATUS_SUCCESS: ClassVar[str] = "success"
    _KEY_ZIP_PATH: ClassVar[str] = "submission_zip"
    _KEY_ADAPTER_PATH: ClassVar[str] = "adapter_path"
    _KEY_ELAPSED_SECONDS: ClassVar[str] = "elapsed_seconds"
    _KEY_ZIP_SIZE_BYTES: ClassVar[str] = "zip_size_bytes"
    _BASE_MODEL_NAME: ClassVar[str] = "gemma-4-31b-it-qat-w4a16-ct"
    _DTYPE_FLOAT32: ClassVar[str] = "float32"
    _STRATEGY_EPOCH: ClassVar[str] = "epoch"
    _SURROGATE_MAX_SEQ_LENGTH: ClassVar[int] = 128
    _SURROGATE_LORA_R: ClassVar[int] = 8
    _SURROGATE_LORA_ALPHA: ClassVar[int] = 16
    _SURROGATE_LR: ClassVar[float] = 1e-4
    _SMOKE_EVAL_FOLD: ClassVar[int] = 1

    def __init__(
        self,
        env_detector: EnvironmentDetector | None = None,
        work_dir: str | None = None,
    ) -> None:
        """Initialize LocalPipelineRunner with optional environment detector and work directory."""
        self._env: EnvironmentDetector = env_detector or EnvironmentDetector()
        self._work_dir: str = work_dir or self._env.get_working_dir()
        self._surrogate_factory: SurrogateModelFactory = SurrogateModelFactory()
        self._validator: ConstraintValidator = ConstraintValidator()
        self._packager: SubmissionPackager = SubmissionPackager(self._validator)

    @classmethod
    def main(cls, argv: list[str] | None = None) -> int:
        """CLI entry point for local pipeline execution."""
        parser = argparse.ArgumentParser(description="SweGemma Local Pipeline Runner")
        parser.add_argument(
            "--mode",
            choices=[cls._MODE_SMOKE, cls._MODE_MOCK, cls._MODE_FULL],
            default=cls._MODE_SMOKE,
            help="Pipeline execution mode (default: smoke)",
        )
        parser.add_argument(
            "--work-dir",
            default=None,
            help="Custom working directory path",
        )
        args = parser.parse_args(argv)
        runner = cls(work_dir=args.work_dir)
        summary = runner.run(mode=args.mode)
        print(json.dumps(summary, indent=2))
        return 0

    def run(self, mode: str = _MODE_SMOKE) -> dict[str, object]:
        """Dispatch pipeline execution based on specified mode.

        Args:
            mode: Pipeline mode ('smoke', 'mock', or 'full').

        Returns:
            Dictionary containing run summary and artifact paths.
        """
        if mode == self._MODE_SMOKE:
            return self.run_smoke()
        if mode == self._MODE_MOCK:
            return self.run_mock()
        if mode == self._MODE_FULL:
            return self._run_full()
        raise ValueError(f"Unknown execution mode: {mode}")

    def run_smoke(self) -> dict[str, object]:
        """Execute real dataset build, SFT training, and packaging with surrogate model."""
        start_time = time.time()
        self._init_workspace()
        surrogate_dir = os.path.join(self._work_dir, self._SUBDIR_SURROGATE)
        self._surrogate_factory.create_surrogate_model(surrogate_dir)
        dataset = DatasetBuilder(mock_mode=True).build_mock()
        sft_config = self._build_smoke_config(surrogate_dir)
        adapter_path = self._execute_sft(sft_config, dataset)
        zip_path = self._package_submission(adapter_path)
        elapsed = time.time() - start_time
        return self._build_summary(zip_path, adapter_path, elapsed)

    def run_mock(self) -> dict[str, object]:
        """Execute lightweight mock pipeline without ML dependencies."""
        start_time = time.time()
        self._init_workspace()
        adapter_dir = os.path.join(
            self._work_dir, self._SUBDIR_CHECKPOINTS, self._ADAPTER_NAME
        )
        self._create_mock_adapter(adapter_dir)
        zip_path = self._package_submission(adapter_dir)
        elapsed = time.time() - start_time
        return self._build_summary(zip_path, adapter_dir, elapsed)

    def _build_smoke_config(self, surrogate_path: str) -> SFTConfig:
        """Build lightweight SFTConfig targeting the surrogate model."""
        checkpoints_dir = os.path.join(self._work_dir, self._SUBDIR_CHECKPOINTS)
        log_file = os.path.join(self._work_dir, self._SUBDIR_LOGS, self._FILE_LOG)
        return SFTConfig(
            model=ModelConfig(
                name=surrogate_path,
                load_in_4bit=False,
                max_seq_length=self._SURROGATE_MAX_SEQ_LENGTH,
                dtype=self._DTYPE_FLOAT32,
            ),
            lora=LoRAConfig(
                r=self._SURROGATE_LORA_R,
                lora_alpha=self._SURROGATE_LORA_ALPHA,
                lora_dropout=0.0,
                target_modules=self._surrogate_factory.get_target_modules(),
            ),
            training=TrainingConfig(
                num_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=self._SURROGATE_LR,
                bf16=False,
                gradient_checkpointing=False,
                max_seq_length=self._SURROGATE_MAX_SEQ_LENGTH,
                packing=False,
                eval_strategy=self._STRATEGY_EPOCH,
                save_strategy=self._STRATEGY_EPOCH,
                load_best_model_at_end=True,
                curriculum_enabled=False,
            ),
            eval_fold=self._SMOKE_EVAL_FOLD,
            output_dir=checkpoints_dir,
            log_path=log_file,
        )

    def _build_summary(
        self, zip_path: str, adapter_path: str, elapsed: float
    ) -> dict[str, object]:
        """Construct execution summary dictionary."""
        zip_size = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
        return {
            self._KEY_STATUS: self._KEY_STATUS_SUCCESS,
            self._KEY_ZIP_PATH: zip_path,
            self._KEY_ADAPTER_PATH: adapter_path,
            self._KEY_ZIP_SIZE_BYTES: zip_size,
            self._KEY_ELAPSED_SECONDS: round(elapsed, 2),
        }

    def _create_mock_adapter(self, adapter_dir: str) -> None:
        """Create mock adapter files that satisfy ConstraintValidator."""
        os.makedirs(adapter_dir, exist_ok=True)
        cfg_path = os.path.join(adapter_dir, self._FILE_ADAPTER_CONFIG)
        cfg_data = {
            "base_model_name_or_path": self._BASE_MODEL_NAME,
            "r": self._SURROGATE_LORA_R,
            "lora_alpha": self._SURROGATE_LORA_ALPHA,
            "target_modules": list(self._surrogate_factory.get_target_modules()),
        }
        with open(cfg_path, "w", encoding="utf-8") as file_handle:
            json.dump(cfg_data, file_handle, indent=2)
        weights_path = os.path.join(adapter_dir, self._FILE_ADAPTER_WEIGHTS)
        with open(weights_path, "wb") as file_handle:
            file_handle.write(b"")

    def _execute_sft(self, sft_config: SFTConfig, dataset: object) -> str:
        """Execute SFT training pipeline and return path to saved adapter."""
        logs_dir = os.path.join(self._work_dir, self._SUBDIR_LOGS)
        telemetry = TelemetryLogger(log_dir=logs_dir, run_version=self._RUN_VERSION)
        checkpoint_mgr = CheckpointManager()
        curriculum = CurriculumScheduler()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)
        return pipeline.run(sft_config, dataset)

    def _init_workspace(self) -> None:
        """Ensure all required workspace directories exist."""
        for subdir in (
            self._SUBDIR_SURROGATE,
            self._SUBDIR_CHECKPOINTS,
            self._SUBDIR_LOGS,
            self._SUBDIR_SUBMISSION,
        ):
            os.makedirs(os.path.join(self._work_dir, subdir), exist_ok=True)

    def _package_submission(self, adapter_path: str) -> str:
        """Package submission zip using templates and trained adapter."""
        deploy_cfg = DeployConfig(
            templates_dir=self._TEMPLATES_DIR,
            output_dir=os.path.join(self._work_dir, self._SUBDIR_SUBMISSION),
            zip_path=os.path.join(self._work_dir, self._FILE_SUBMISSION_ZIP),
            adapters=[{"name": self._ADAPTER_NAME, "checkpoint_path": adapter_path}],
        )
        return self._packager.package(deploy_cfg)

    def _run_full(self) -> dict[str, object]:
        """Execute full training pipeline using default configuration."""
        raise NotImplementedError("Full training pipeline requires Kaggle GPU cluster")


if __name__ == "__main__":
    LocalPipelineRunner.main()
