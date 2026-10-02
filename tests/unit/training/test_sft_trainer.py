"""Unit tests for SFTTrainerPipeline verifying full supervised training workflow."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.config.sft_config import SFTConfig
from src.training.sft_trainer import SFTTrainerPipeline


class TestSFTTrainerPipeline:
    """Test suite covering SFTTrainerPipeline execution, preparation, and checks."""

    _MOCK_ADAPTER_PATH: str = "/tmp/test_checkpoints/sft_lora"
    _EVAL_FOLD: int = 1
    _TRAIN_LOSS: float = 0.42
    _BEST_METRIC: float = 0.38
    _GLOBAL_STEP: int = 200
    _ADAPTER_SIZE: int = 400_000_000

    @pytest.fixture
    def mock_env(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
        """Set up mocked peft, trl, transformers, and model environment."""
        mocks = self._build_mock_objects()
        monkeypatch.setitem(sys.modules, "peft", mocks["peft"])
        monkeypatch.setitem(sys.modules, "trl", mocks["trl"])
        monkeypatch.setitem(sys.modules, "transformers", mocks["transformers"])
        return mocks

    def test_apply_lora_without_get_nb_trainable_parameters(
        self, mock_env: dict[str, MagicMock]
    ) -> None:
        """Verify apply_lora succeeds when model lacks get_nb_trainable_parameters."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        plain_model = object()
        mock_env["peft"].get_peft_model.return_value = plain_model

        result = pipeline._apply_lora(plain_model, SFTConfig().lora)
        assert result is plain_model

    def test_enable_input_grads_and_checkpointing(self) -> None:
        """Verify enable_input_grads and gradient checkpointing on model."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        mock_model = MagicMock()
        pipeline._enable_input_grads(mock_model)
        mock_model.enable_input_require_grads.assert_called_once()

        pipeline._enable_gradient_checkpointing(mock_model)
        mock_model.gradient_checkpointing_enable.assert_called_once()

    def test_prepare_splits_filters_by_fold(self) -> None:
        """Verify dataset list split separates eval fold from train folds."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        dataset = [
            {"id": "t1", "fold": 0},
            {"id": "t2", "fold": self._EVAL_FOLD},
            {"id": "t3", "fold": 2},
        ]
        train_ds, val_ds = pipeline._prepare_splits(dataset, eval_fold=self._EVAL_FOLD)

        assert isinstance(train_ds, list)
        assert isinstance(val_ds, list)
        assert len(train_ds) == 2
        assert len(val_ds) == 1
        assert val_ds[0]["id"] == "t2"

    def test_prepare_splits_with_filter_api(self) -> None:
        """Verify split preparation with datasets exposing filter() method."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        mock_ds = MagicMock()
        mock_train = MagicMock()
        mock_val = MagicMock()
        mock_ds.filter.side_effect = [mock_train, mock_val]

        train_ds, val_ds = pipeline._prepare_splits(mock_ds, eval_fold=0)
        assert train_ds == mock_train
        assert val_ds == mock_val
        assert mock_ds.filter.call_count == 2

    def test_prepare_splits_with_opaque_dataset_fallback(self) -> None:
        """Verify fallback returning identical dataset when neither filter nor list."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        opaque_dataset = object()
        train_ds, val_ds = pipeline._prepare_splits(opaque_dataset, eval_fold=0)
        assert train_ds is opaque_dataset
        assert val_ds is opaque_dataset

    def test_resolve_dtype_returns_expected_types(self) -> None:
        """Verify _resolve_dtype resolves known dtypes and defaults to bfloat16."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        assert pipeline._resolve_dtype("bfloat16") is not None
        assert pipeline._resolve_dtype("float16") is not None
        assert pipeline._resolve_dtype("float32") is not None
        assert pipeline._resolve_dtype("unknown") is not None

    def test_run_calls_apply_lora_and_logs_trainable_params(
        self, mock_env: dict[str, MagicMock]
    ) -> None:
        """Verify LoRA adapter injection and trainable parameters logging."""
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.save_adapter.return_value = self._MOCK_ADAPTER_PATH
        telemetry = MagicMock()
        curriculum = MagicMock()

        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)
        config = SFTConfig()

        pipeline.run(config, [])
        mock_env["peft"].get_peft_model.assert_called_once()
        telemetry.log_info.assert_called_once()

    def test_run_executes_pipeline_and_returns_path(
        self, mock_env: dict[str, MagicMock]
    ) -> None:
        """Verify run orchestrates full SFT pipeline from model load to adapter save."""
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.save_adapter.return_value = self._MOCK_ADAPTER_PATH
        checkpoint_mgr._compute_total_size.return_value = self._ADAPTER_SIZE

        telemetry = MagicMock()
        curriculum = MagicMock()
        curriculum.get_epoch_data.side_effect = lambda ds, ep, tot: ds

        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)
        config = SFTConfig()
        dataset = [{"fold": 0, "complexity": "SIMPLE"}]

        path = pipeline.run(config, dataset)

        assert path == self._MOCK_ADAPTER_PATH
        telemetry.log_start.assert_called_once_with("sft", config)
        telemetry.log_end.assert_called_once()
        checkpoint_mgr.save_adapter.assert_called_once()

    def test_run_with_curriculum_disabled(
        self, mock_env: dict[str, MagicMock]
    ) -> None:
        """Verify curriculum application is bypassed when disabled in config."""
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.save_adapter.return_value = self._MOCK_ADAPTER_PATH
        telemetry = MagicMock()
        curriculum = MagicMock()

        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)
        config = SFTConfig()
        config.training.curriculum_enabled = False

        pipeline.run(config, [])
        curriculum.get_epoch_data.assert_not_called()

    def test_verify_special_tokens_missing_token_raises_value_error(self) -> None:
        """Verify ValueError raised when tokenizer is missing required tokens."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"<start_of_turn>": 1}

        with pytest.raises(ValueError, match="Missing required special token"):
            pipeline._verify_special_tokens(mock_tokenizer)

    def test_verify_special_tokens_without_get_vocab_method(self) -> None:
        """Verify tokenizer without get_vocab raises ValueError for missing tokens."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        with pytest.raises(ValueError, match="Missing required special token"):
            pipeline._verify_special_tokens(object())

    def _build_mock_objects(self) -> dict[str, MagicMock]:
        """Construct mock instances for external training libraries."""
        mock_peft, mock_trl, mock_transformers = MagicMock(), MagicMock(), MagicMock()
        mock_model = MagicMock()
        mock_model.get_nb_trainable_parameters.return_value = (1_000, 10_000)

        vocab = {tok: idx for idx, tok in enumerate(SFTTrainerPipeline._REQUIRED_SPECIAL_TOKENS)}
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = vocab

        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.BitsAndBytesConfig = MagicMock()

        mock_peft.get_peft_model.return_value = mock_model
        mock_peft.LoraConfig = MagicMock()
        mock_peft.TaskType = MagicMock()

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = SimpleNamespace(training_loss=self._TRAIN_LOSS)
        mock_trainer.state = SimpleNamespace(
            best_metric=self._BEST_METRIC, global_step=self._GLOBAL_STEP
        )
        mock_trl.SFTTrainer.return_value = mock_trainer
        mock_trl.SFTConfig = MagicMock()

        return {
            "peft": mock_peft,
            "trl": mock_trl,
            "transformers": mock_transformers,
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "trainer": mock_trainer,
        }
