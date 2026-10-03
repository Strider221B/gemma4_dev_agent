"""Unit tests for RLTrainerPipeline orchestrating GRPO and DPO reinforcement learning."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.config.rl_config import RLConfig
from src.data.task import Task
from src.training.checkpoint_manager import CheckpointManager
from src.training.reward_model import RewardModel
from src.training.rl_trainer import RLTrainerPipeline
from src.utils.telemetry_logger import TelemetryLogger


class TestRLTrainerPipeline:
    """Test suite verifying RLTrainerPipeline execution, model preparation, and training modes."""

    _MOCK_ADAPTER_PATH: str = "/checkpoints/rl_lora"
    _INSTANCE_ID: str = "task_rl_01"
    _REPO: str = "google/gemma4"
    _COMMIT: str = "c0ffee"
    _PROBLEM: str = "Fix memory leak"
    _HINTS: str = "Look at GC"
    _PATCH: str = "diff --git a/a.py b/a.py"
    _TEST_PATCH: str = "diff --git a/test_a.py b/test_a.py"
    _CREATED_AT: str = "2026-10-01T00:00:00Z"
    _MOCK_SIZE_BYTES: int = 500_000
    _MOCK_RESOLVED_PATH: str = "/resolved/rl/model"
    _RESOLVED_TARGETS: list[str] = ["q_proj.linear", "v_proj.linear"]
    _KEY_TARGET_MODULES: str = "target_modules"

    @pytest.fixture
    def sample_task(self) -> Task:
        """Create sample Task object for pipeline tests."""
        return Task(
            instance_id=self._INSTANCE_ID,
            repo=self._REPO,
            base_commit=self._COMMIT,
            problem_statement=self._PROBLEM,
            hints_text=self._HINTS,
            patch=self._PATCH,
            test_patch=self._TEST_PATCH,
            created_at=self._CREATED_AT,
        )

    @pytest.fixture
    def pipeline_fixture(
        self,
    ) -> tuple[RLTrainerPipeline, MagicMock, MagicMock, MagicMock]:
        """Create RLTrainerPipeline instance with mock dependencies."""
        mock_reward = MagicMock(spec=RewardModel)
        mock_ckpt = MagicMock(spec=CheckpointManager)
        mock_telemetry = MagicMock(spec=TelemetryLogger)
        pipeline = RLTrainerPipeline(mock_reward, mock_ckpt, mock_telemetry)
        return pipeline, mock_reward, mock_ckpt, mock_telemetry

    def test_run_grpo_calls_expected_methods(
        self,
        sample_task: Task,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify pipeline execution in GRPO mode calls training and returns adapter path."""
        pipeline, _, _, mock_telemetry = pipeline_fixture
        pipeline._load_sft_model = MagicMock(return_value=(MagicMock(), MagicMock()))  # type: ignore[method-assign]
        pipeline._merge_and_reapply_lora = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
        pipeline._train_grpo = MagicMock(return_value=self._MOCK_ADAPTER_PATH)  # type: ignore[method-assign]
        pipeline._compute_size = MagicMock(return_value=self._MOCK_SIZE_BYTES)  # type: ignore[method-assign]

        config = RLConfig(mode="grpo")
        result = pipeline.run(config, [sample_task])

        assert result == self._MOCK_ADAPTER_PATH
        pipeline._train_grpo.assert_called_once()
        mock_telemetry.log_start.assert_called_once_with("rl", config)
        mock_telemetry.log_end.assert_called_once()

    def test_run_dpo_calls_expected_methods(
        self,
        sample_task: Task,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify pipeline execution in DPO mode calls DPO training flow."""
        pipeline, _, _, mock_telemetry = pipeline_fixture
        pipeline._load_sft_model = MagicMock(return_value=(MagicMock(), MagicMock()))  # type: ignore[method-assign]
        pipeline._merge_and_reapply_lora = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
        pipeline._train_dpo = MagicMock(return_value=self._MOCK_ADAPTER_PATH)  # type: ignore[method-assign]
        pipeline._compute_size = MagicMock(return_value=self._MOCK_SIZE_BYTES)  # type: ignore[method-assign]

        config = RLConfig(mode="dpo")
        result = pipeline.run(config, [sample_task])

        assert result == self._MOCK_ADAPTER_PATH
        pipeline._train_dpo.assert_called_once()
        mock_telemetry.log_start.assert_called_once_with("rl", config)
        mock_telemetry.log_end.assert_called_once()

    def test_run_invalid_mode_raises_error(
        self,
        sample_task: Task,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify invalid training mode raises ValueError."""
        pipeline, _, _, _ = pipeline_fixture
        pipeline._load_sft_model = MagicMock(return_value=(MagicMock(), MagicMock()))  # type: ignore[method-assign]
        pipeline._merge_and_reapply_lora = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]

        config = RLConfig()
        config.mode = "invalid_mode"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown RL mode: invalid_mode"):
            pipeline.run(config, [sample_task])

    def test_build_prompts_creates_dataset_records(
        self,
        sample_task: Task,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify build_prompts formats task prompts with proper turn delimiters."""
        pipeline, _, _, _ = pipeline_fixture
        prompts = pipeline._build_prompts([sample_task])
        assert len(prompts) == 1
        prompt_text = (
            prompts[0]["prompt"]
            if isinstance(prompts[0], dict)
            else prompts["prompt"][0]
        )
        assert "<|turn>user" in prompt_text
        assert "<turn|>" in prompt_text
        assert sample_task.problem_statement in prompt_text

    def test_compute_size_delegates_to_checkpoint_manager(
        self,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify _compute_size delegates directory size inquiry to CheckpointManager."""
        pipeline, _, mock_ckpt, _ = pipeline_fixture
        mock_ckpt._compute_total_size.return_value = 12345
        size = pipeline._compute_size(self._MOCK_ADAPTER_PATH)
        assert size == 12345
        mock_ckpt._compute_total_size.assert_called_once_with(
            self._MOCK_ADAPTER_PATH
        )

    def test_enable_input_grads_and_checkpointing(
        self,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify enable_input_grads and gradient checkpointing on model."""
        pipeline, _, _, _ = pipeline_fixture
        mock_model = MagicMock()
        pipeline._enable_input_grads(mock_model)
        mock_model.enable_input_require_grads.assert_called_once()
        pipeline._enable_gradient_checkpointing(mock_model)
        mock_model.gradient_checkpointing_enable.assert_called_once()

    def test_load_sft_model_and_reapply_lora(
        self,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify SFT model loading and LoRA re-application using transformers and peft."""
        pipeline, _, _, mock_telemetry = pipeline_fixture
        mocks = self._setup_model_mocks()
        with patch.dict("sys.modules", mocks["modules"]):
            config = RLConfig()
            model, tokenizer = pipeline._load_sft_model(config)
            assert model == mocks["model"]
            assert tokenizer == mocks["tokenizer"]
            mock_telemetry.log_info.assert_called_once()
            call_kwargs = (
                mocks["transformers"].AutoModelForCausalLM.from_pretrained.call_args.kwargs
            )
            assert "quantization_config" in call_kwargs

            lora_model = pipeline._merge_and_reapply_lora(model, config)
            assert lora_model == mocks["model"]

    def test_merge_and_reapply_lora_resolves_clippable_targets(
        self,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify _merge_and_reapply_lora resolves targets and passes to PeftLoraConfig."""
        pipeline, _, _, _ = pipeline_fixture
        mocks = self._setup_model_mocks()
        with patch.dict("sys.modules", mocks["modules"]):
            config = RLConfig()
            with patch(
                "src.training.lora_target_resolver.LoRATargetModuleResolver.resolve"
            ) as mock_resolve:
                mock_resolve.return_value = self._RESOLVED_TARGETS
                lora_model = pipeline._merge_and_reapply_lora(mocks["model"], config)
                assert lora_model == mocks["model"]
                mock_resolve.assert_called_once_with(
                    mocks["model"], list(pipeline._TARGET_MODULES)
                )
                call_kwargs = mocks["peft"].LoraConfig.call_args.kwargs
                assert call_kwargs[self._KEY_TARGET_MODULES] == self._RESOLVED_TARGETS

    def test_load_sft_model_resolves_path(
        self,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify _load_sft_model resolves model path via ModelPathResolver."""
        pipeline, _, _, _ = pipeline_fixture
        mocks = self._setup_model_mocks()
        with patch.dict("sys.modules", mocks["modules"]):
            with patch(
                "src.utils.model_path_resolver.ModelPathResolver.resolve",
                return_value=self._MOCK_RESOLVED_PATH,
            ) as mock_res:
                pipeline._load_sft_model(RLConfig())
                mock_res.assert_called_once()
                call_args = mocks["transformers"].AutoModelForCausalLM.from_pretrained.call_args[0]
                assert call_args[0] == self._MOCK_RESOLVED_PATH

    def test_load_sft_model_skips_quantization_for_prequantized_model(
        self,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify _load_sft_model omits BitsAndBytesConfig when native quantization exists."""
        pipeline, _, _, _ = pipeline_fixture
        mocks = self._setup_model_mocks()
        mocks["transformers"].AutoConfig.from_pretrained.return_value.quantization_config = (
            MagicMock()
        )
        with patch.dict("sys.modules", mocks["modules"]):
            with patch(
                "src.utils.model_path_resolver.ModelPathResolver.resolve",
                return_value=self._MOCK_RESOLVED_PATH,
            ):
                pipeline._load_sft_model(RLConfig())
                call_kwargs = (
                    mocks["transformers"].AutoModelForCausalLM.from_pretrained.call_args.kwargs
                )
                assert "quantization_config" not in call_kwargs

    def test_load_sft_model_skips_quantization_when_load_in_4bit_false(
        self,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify _load_sft_model omits BitsAndBytesConfig when load_in_4bit is False."""
        pipeline, _, _, _ = pipeline_fixture
        mocks = self._setup_model_mocks()
        with patch.dict("sys.modules", mocks["modules"]):
            with patch(
                "src.utils.model_path_resolver.ModelPathResolver.resolve",
                return_value=self._MOCK_RESOLVED_PATH,
            ):
                config = RLConfig()
                config.model.load_in_4bit = False
                pipeline._load_sft_model(config)
                call_kwargs = (
                    mocks["transformers"].AutoModelForCausalLM.from_pretrained.call_args.kwargs
                )
                assert "quantization_config" not in call_kwargs

    def test_resolve_dtype_returns_expected_types(
        self,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify _resolve_dtype resolves known dtypes and handles ImportError."""
        pipeline, _, _, _ = pipeline_fixture
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert pipeline._resolve_dtype("bfloat16") is not None
            assert pipeline._resolve_dtype("float16") is not None
            assert pipeline._resolve_dtype("float32") is not None
            assert pipeline._resolve_dtype("unknown") is not None

        with patch.dict("sys.modules", {"torch": None}):
            assert pipeline._resolve_dtype("bfloat16") == "bfloat16"

    def test_train_grpo_instantiates_trainer(
        self,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify _train_grpo constructs GRPOTrainer and saves adapter."""
        pipeline, _, mock_ckpt, _ = pipeline_fixture
        mock_ckpt.save_adapter.return_value = self._MOCK_ADAPTER_PATH
        mock_trl = MagicMock()
        mock_trainer_cls = MagicMock()
        mock_trainer_inst = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_inst
        mock_trl.GRPOTrainer = mock_trainer_cls
        mock_trl.GRPOConfig = MagicMock()

        with patch.dict("sys.modules", {"trl": mock_trl}):
            config = RLConfig()
            adapter_path = pipeline._train_grpo(
                MagicMock(), MagicMock(), [], config
            )
            assert adapter_path == self._MOCK_ADAPTER_PATH
            mock_trainer_inst.train.assert_called_once()
            mock_ckpt.save_adapter.assert_called_once()

    def test_train_dpo_instantiates_trainer(
        self,
        pipeline_fixture: tuple[
            RLTrainerPipeline, MagicMock, MagicMock, MagicMock
        ],
    ) -> None:
        """Verify _train_dpo constructs DPOTrainer and saves adapter."""
        pipeline, _, mock_ckpt, _ = pipeline_fixture
        mock_ckpt.save_adapter.return_value = self._MOCK_ADAPTER_PATH
        mock_trl = MagicMock()
        mock_trainer_cls = MagicMock()
        mock_trainer_inst = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_inst
        mock_trl.DPOTrainer = mock_trainer_cls
        mock_trl.DPOConfig = MagicMock()

        with patch.dict("sys.modules", {"trl": mock_trl}):
            config = RLConfig()
            adapter_path = pipeline._train_dpo(
                MagicMock(), MagicMock(), [], config
            )
            assert adapter_path == self._MOCK_ADAPTER_PATH
            mock_trainer_inst.train.assert_called_once()
            mock_ckpt.save_adapter.assert_called_once()

    def _setup_model_mocks(self) -> dict[str, Any]:
        """Construct mock instances for transformers, peft, and torch."""
        mock_transformers, mock_peft, mock_torch = MagicMock(), MagicMock(), MagicMock()
        mock_model, mock_tokenizer = MagicMock(), MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoConfig.from_pretrained.return_value.quantization_config = None
        mock_transformers.BitsAndBytesConfig = MagicMock()
        mock_peft_inst = MagicMock()
        mock_peft_inst.merge_and_unload.return_value = mock_model
        mock_peft.PeftModel.from_pretrained.return_value = mock_peft_inst
        mock_peft.get_peft_model.return_value = mock_model
        return {
            "modules": {
                "transformers": mock_transformers,
                "peft": mock_peft,
                "torch": mock_torch,
            },
            "transformers": mock_transformers,
            "peft": mock_peft,
            "model": mock_model,
            "tokenizer": mock_tokenizer,
        }
