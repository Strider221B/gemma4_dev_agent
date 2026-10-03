"""Unit tests for SFTTrainerPipeline verifying full supervised training workflow."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    _MOCK_RESOLVED_PATH: str = "/resolved/model/path"
    _RESOLVED_TARGETS: list[str] = ["q_proj.linear", "v_proj.linear"]
    _KEY_TARGET_MODULES: str = "target_modules"

    @pytest.fixture
    def mock_env(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
        """Set up mocked peft, trl, transformers, and model environment."""
        mocks = self._build_mock_objects()
        monkeypatch.setitem(sys.modules, "peft", mocks["peft"])
        monkeypatch.setitem(sys.modules, "trl", mocks["trl"])
        monkeypatch.setitem(sys.modules, "transformers", mocks["transformers"])
        monkeypatch.setitem(sys.modules, "torch", mocks["torch"])
        return mocks

    def test_apply_lora_resolves_clippable_targets(
        self, mock_env: dict[str, MagicMock]
    ) -> None:
        """Verify _apply_lora resolves target modules and passes to PeftLoraConfig."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        mock_model = mock_env["model"]
        mock_env["peft"].get_peft_model.return_value = mock_model
        config = SFTConfig()

        with patch(
            "src.training.lora_target_resolver.LoRATargetModuleResolver.resolve"
        ) as mock_resolve:
            mock_resolve.return_value = self._RESOLVED_TARGETS
            result = pipeline._apply_lora(mock_model, config.lora)

            assert result is mock_model
            mock_resolve.assert_called_once_with(mock_model, config.lora.target_modules)
            call_kwargs = mock_env["peft"].LoraConfig.call_args.kwargs
            assert call_kwargs[self._KEY_TARGET_MODULES] == self._RESOLVED_TARGETS

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
        """Verify _resolve_dtype resolves known dtypes and handles ImportError."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        mock_torch = MagicMock()
        with patch.dict(sys.modules, {"torch": mock_torch}):
            assert pipeline._resolve_dtype("bfloat16") is not None
            assert pipeline._resolve_dtype("float16") is not None
            assert pipeline._resolve_dtype("float32") is not None
            assert pipeline._resolve_dtype("unknown") is not None

        with patch.dict(sys.modules, {"torch": None}):
            assert pipeline._resolve_dtype("bfloat16") == "bfloat16"

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
        mock_tokenizer.get_vocab.return_value = {"<|turn>": 1}

        with pytest.raises(ValueError, match="Missing required special token"):
            pipeline._verify_special_tokens(mock_tokenizer)

    def test_verify_special_tokens_with_all_tokens_succeeds(self) -> None:
        """Verify _verify_special_tokens passes when all Gemma 4 tokens are present."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {
            tok: idx for idx, tok in enumerate(SFTTrainerPipeline._REQUIRED_SPECIAL_TOKENS)
        }
        pipeline._verify_special_tokens(mock_tokenizer)

    def test_verify_special_tokens_without_get_vocab_method(self) -> None:
        """Verify tokenizer without get_vocab raises ValueError for missing tokens."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        with pytest.raises(ValueError, match="Missing required special token"):
            pipeline._verify_special_tokens(object())

    def test_load_model_resolves_path(self, mock_env: dict[str, MagicMock]) -> None:
        """Verify _load_model uses ModelPathResolver to determine model path."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)
        config = SFTConfig()

        with patch("src.utils.model_path_resolver.ModelPathResolver.resolve") as mock_res:
            mock_res.return_value = self._MOCK_RESOLVED_PATH
            pipeline._load_model(config.model)
            mock_res.assert_called_once_with(config.model.name)
            mock_env["transformers"].AutoModelForCausalLM.from_pretrained.assert_called_with(
                self._MOCK_RESOLVED_PATH,
                quantization_config=mock_env["transformers"].BitsAndBytesConfig.return_value,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=mock_env["torch"].bfloat16,
            )

    def test_load_model_skips_quantization_for_prequantized_model(
        self, mock_env: dict[str, MagicMock]
    ) -> None:
        """Verify _load_model omits BitsAndBytesConfig when model has native quantization."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)
        config = SFTConfig()
        mock_env["transformers"].AutoConfig.from_pretrained.return_value.quantization_config = (
            MagicMock()
        )

        with patch("src.utils.model_path_resolver.ModelPathResolver.resolve") as mock_res:
            mock_res.return_value = self._MOCK_RESOLVED_PATH
            pipeline._load_model(config.model)
            call_kwargs = (
                mock_env["transformers"].AutoModelForCausalLM.from_pretrained.call_args.kwargs
            )
            assert "quantization_config" not in call_kwargs

    def test_load_model_skips_quantization_when_load_in_4bit_false(
        self, mock_env: dict[str, MagicMock]
    ) -> None:
        """Verify _load_model omits BitsAndBytesConfig when load_in_4bit is False."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)
        config = SFTConfig()
        config.model.load_in_4bit = False

        with patch("src.utils.model_path_resolver.ModelPathResolver.resolve") as mock_res:
            mock_res.return_value = self._MOCK_RESOLVED_PATH
            pipeline._load_model(config.model)
            call_kwargs = (
                mock_env["transformers"].AutoModelForCausalLM.from_pretrained.call_args.kwargs
            )
            assert "quantization_config" not in call_kwargs

    def test_create_trl_config_direct_success(self) -> None:
        """Verify _create_trl_config instantiates class directly when args match."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        mock_cls = MagicMock(return_value="config_instance")
        result = pipeline._create_trl_config(mock_cls, {"max_length": 512})
        assert result == "config_instance"
        mock_cls.assert_called_once_with(max_length=512)

    def test_create_trl_config_fallback_max_seq_length(self) -> None:
        """Verify fallback from max_length to max_seq_length on TypeError."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        mock_cls = MagicMock()
        mock_cls.side_effect = [
            TypeError("SFTConfig.__init__() got an unexpected keyword argument 'max_length'"),
            "fallback_instance",
        ]
        result = pipeline._create_trl_config(mock_cls, {"max_length": 512, "epochs": 3})
        assert result == "fallback_instance"
        assert mock_cls.call_count == 2
        mock_cls.assert_called_with(max_seq_length=512, epochs=3)

    def test_create_trl_config_fallback_max_length(self) -> None:
        """Verify fallback from max_seq_length to max_length on TypeError."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        mock_cls = MagicMock()
        mock_cls.side_effect = [
            TypeError("SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'"),
            "fallback_instance",
        ]
        result = pipeline._create_trl_config(mock_cls, {"max_seq_length": 512})
        assert result == "fallback_instance"
        assert mock_cls.call_count == 2
        mock_cls.assert_called_with(max_length=512)

    def test_create_trl_config_unrelated_type_error_raises(self) -> None:
        """Verify unexpected TypeError not involving seq length is re-raised."""
        checkpoint_mgr = MagicMock()
        telemetry = MagicMock()
        curriculum = MagicMock()
        pipeline = SFTTrainerPipeline(checkpoint_mgr, telemetry, curriculum)

        mock_cls = MagicMock(
            side_effect=TypeError("unexpected keyword argument 'invalid_param'")
        )
        with pytest.raises(TypeError, match="invalid_param"):
            pipeline._create_trl_config(mock_cls, {"invalid_param": 1})

    def _build_mock_objects(self) -> dict[str, MagicMock]:
        """Construct mock instances for external training libraries."""
        mock_peft, mock_trl, mock_transformers = MagicMock(), MagicMock(), MagicMock()
        mock_torch = MagicMock()
        mock_model = MagicMock()
        mock_model.get_nb_trainable_parameters.return_value = (1_000, 10_000)

        vocab = {tok: idx for idx, tok in enumerate(SFTTrainerPipeline._REQUIRED_SPECIAL_TOKENS)}
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = vocab

        mock_model_config = MagicMock()
        mock_model_config.quantization_config = None
        mock_transformers.AutoConfig.from_pretrained.return_value = mock_model_config
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
            "torch": mock_torch,
            "model": mock_model,
            "tokenizer": mock_tokenizer,
            "trainer": mock_trainer,
        }
