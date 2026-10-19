"""Unit tests for LocalModelSmokeTester executing with zero physical weights."""

from __future__ import annotations

import tempfile
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from src.training.local_model_smoke_tester import LocalModelSmokeTester
from src.training.model_verification_runner import ModelVerificationRunner
from src.utils.model_path_resolver import ModelPathResolver


class TestLocalModelSmokeTester:
    """Unit test suite for LocalModelSmokeTester verification workflow."""

    _CUSTOM_MODEL_NAME: ClassVar[str] = "custom-org/custom-gemma"
    _MOCK_RESOLVED_PATH: ClassVar[str] = "/mock/path/gemma-model"
    _STATUS_KEY: ClassVar[str] = "status"
    _STATUS_SUCCESS: ClassVar[str] = "success"
    _FORWARD_KEY: ClassVar[str] = "forward_passed"
    _BACKWARD_KEY: ClassVar[str] = "backward_passed"
    _ADAPTER_SAVED_KEY: ClassVar[str] = "adapter_saved"
    _MISSING_TOKEN_MSG: ClassVar[str] = "Missing required special token"
    _NON_CALLABLE_MSG: ClassVar[str] = "Model object is not callable"
    _NO_LOGITS_MSG: ClassVar[str] = "Model outputs do not contain logits or loss"
    _ARG_MODEL_NAME: ClassVar[str] = "--model-name"
    _ARG_WORK_DIR: ClassVar[str] = "--work-dir"

    def test_implements_model_verification_runner_protocol(self) -> None:
        """Verify LocalModelSmokeTester satisfies ModelVerificationRunner protocol."""
        tester = LocalModelSmokeTester()
        assert isinstance(tester, ModelVerificationRunner)

    def test_instantiation_with_defaults_and_custom_args(self) -> None:
        """Verify instantiation properly assigns defaults and explicit parameter overrides."""
        tester_default = LocalModelSmokeTester()
        assert tester_default._model_name == LocalModelSmokeTester._DEFAULT_MODEL_NAME

        with tempfile.TemporaryDirectory() as tmpdir:
            tester_custom = LocalModelSmokeTester(
                model_name=self._CUSTOM_MODEL_NAME, work_dir=tmpdir
            )
            assert tester_custom._model_name == self._CUSTOM_MODEL_NAME
            assert tester_custom._work_dir == tmpdir

    def test_resolve_model_path_delegates_to_model_path_resolver(self) -> None:
        """Verify _resolve_model_path calls ModelPathResolver.resolve with target model name."""
        tester = LocalModelSmokeTester(model_name=self._CUSTOM_MODEL_NAME)
        with patch.object(
            ModelPathResolver, "resolve", return_value=self._MOCK_RESOLVED_PATH
        ) as mock_resolve:
            resolved = tester._resolve_model_path()
            assert resolved == self._MOCK_RESOLVED_PATH
            mock_resolve.assert_called_once_with(self._CUSTOM_MODEL_NAME)

    def test_resolve_model_device_returns_device_when_present(self) -> None:
        """Verify _resolve_model_device extracts device string or torch.device."""
        import torch

        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([mock_param])

        tester = LocalModelSmokeTester()
        dev = tester._resolve_model_device(mock_model)
        assert dev == "cpu"

    def test_resolve_model_device_returns_none_when_no_parameters(self) -> None:
        """Verify _resolve_model_device returns None when model lacks parameters."""
        tester = LocalModelSmokeTester()
        assert tester._resolve_model_device(object()) is None

    def test_verify_tokenizer_tokens_succeeds_when_all_tokens_present(self) -> None:
        """Verify _verify_tokenizer_tokens completes when all required tokens exist in vocab."""
        mock_tokenizer = MagicMock()
        mock_vocab = {
            token: idx
            for idx, token in enumerate(LocalModelSmokeTester._REQUIRED_SPECIAL_TOKENS)
        }
        mock_tokenizer.get_vocab.return_value = mock_vocab

        tester = LocalModelSmokeTester()
        tester._verify_tokenizer_tokens(mock_tokenizer)
        mock_tokenizer.get_vocab.assert_called_once()

    def test_verify_tokenizer_tokens_raises_error_when_token_missing(self) -> None:
        """Verify _verify_tokenizer_tokens raises ValueError if any required token is absent."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab.return_value = {"<|turn>": 1}

        tester = LocalModelSmokeTester()
        with pytest.raises(ValueError, match=self._MISSING_TOKEN_MSG):
            tester._verify_tokenizer_tokens(mock_tokenizer)

    def test_attach_lora_adapters_invokes_patcher_and_resolver(self) -> None:
        """Verify _attach_lora_adapters patches class, resolves targets, and patches model."""
        mock_patcher = MagicMock()
        mock_resolver = MagicMock()
        mock_resolver.resolve.return_value = ["q_proj", "v_proj"]
        mock_model = MagicMock()
        mock_peft_model = MagicMock()

        tester = LocalModelSmokeTester(patcher=mock_patcher, target_resolver=mock_resolver)
        with patch("peft.get_peft_model", return_value=mock_peft_model) as mock_get_peft:
            result = tester._attach_lora_adapters(mock_model)
            assert result == mock_patcher.patch_model.return_value
            mock_patcher.patch_peft_lora_class.assert_called_once()
            mock_resolver.resolve.assert_called_once()
            mock_get_peft.assert_called_once()
            mock_patcher.patch_model.assert_called_once_with(mock_peft_model)

    def test_execute_forward_pass_executes_callable_model(self) -> None:
        """Verify _execute_forward_pass passes input_ids and attention_mask to model."""
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 0

        tester = LocalModelSmokeTester()
        outputs = tester._execute_forward_pass(mock_model, mock_tokenizer)
        assert outputs == mock_outputs
        mock_model.assert_called_once()

    def test_execute_forward_pass_raises_error_when_model_not_callable(self) -> None:
        """Verify _execute_forward_pass raises TypeError when model is not callable."""
        tester = LocalModelSmokeTester()
        mock_tokenizer = MagicMock()
        with pytest.raises(TypeError, match=self._NON_CALLABLE_MSG):
            tester._execute_forward_pass(object(), mock_tokenizer)

    def test_execute_backward_pass_computes_loss_and_calls_backward(self) -> None:
        """Verify _execute_backward_pass triggers backward() on loss tensor."""
        mock_loss = MagicMock()
        mock_outputs = MagicMock()
        mock_outputs.loss = mock_loss

        tester = LocalModelSmokeTester()
        success = tester._execute_backward_pass(mock_outputs)
        assert success is True
        mock_loss.backward.assert_called_once()

    def test_execute_backward_pass_computes_loss_from_logits_when_loss_none(self) -> None:
        """Verify _execute_backward_pass sums logits and backpropagates when loss is None."""
        mock_logits = MagicMock()
        mock_loss = MagicMock()
        mock_logits.sum.return_value = mock_loss
        mock_outputs = MagicMock()
        mock_outputs.loss = None
        mock_outputs.logits = mock_logits

        tester = LocalModelSmokeTester()
        success = tester._execute_backward_pass(mock_outputs)
        assert success is True
        mock_logits.sum.assert_called_once()
        mock_loss.backward.assert_called_once()

    def test_execute_backward_pass_raises_error_when_no_logits_or_loss(self) -> None:
        """Verify _execute_backward_pass raises AttributeError when output lacks logits."""
        mock_outputs = MagicMock()
        mock_outputs.loss = None
        mock_outputs.logits = None

        tester = LocalModelSmokeTester()
        with pytest.raises(AttributeError, match=self._NO_LOGITS_MSG):
            tester._execute_backward_pass(mock_outputs)

    def test_save_adapter_checkpoint_invokes_save_pretrained(self) -> None:
        """Verify _save_adapter_checkpoint delegates to model save_pretrained."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_model = MagicMock()
            tester = LocalModelSmokeTester(work_dir=tmpdir)
            adapter_dir = tester._save_adapter_checkpoint(mock_model)
            assert adapter_dir.startswith(tmpdir)
            mock_model.save_pretrained.assert_called_once_with(adapter_dir)

    def test_run_verification_end_to_end_mocked_succeeds(self) -> None:
        """Verify run_verification orchestrates all 7 stages and returns success report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tester = LocalModelSmokeTester(work_dir=tmpdir)
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_outputs = MagicMock()

            with (
                patch.object(tester, "_resolve_model_path", return_value=self._MOCK_RESOLVED_PATH),
                patch.object(
                    tester, "_load_model_and_tokenizer", return_value=(mock_model, mock_tokenizer)
                ),
                patch.object(tester, "_verify_tokenizer_tokens"),
                patch.object(tester, "_attach_lora_adapters", return_value=mock_model),
                patch.object(tester, "_execute_forward_pass", return_value=mock_outputs),
                patch.object(tester, "_execute_backward_pass", return_value=True),
                patch.object(tester, "_save_adapter_checkpoint", return_value=tmpdir),
            ):
                report = tester.run_verification()
                assert report[self._STATUS_KEY] == self._STATUS_SUCCESS
                assert report[self._FORWARD_KEY] is True
                assert report[self._BACKWARD_KEY] is True
                assert report[self._ADAPTER_SAVED_KEY] is True

    def test_cli_main_parses_arguments_and_executes(self) -> None:
        """Verify class main method parses CLI flags and returns status 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = [
                self._ARG_MODEL_NAME,
                self._CUSTOM_MODEL_NAME,
                self._ARG_WORK_DIR,
                tmpdir,
            ]
            mock_summary = {self._STATUS_KEY: self._STATUS_SUCCESS}
            with patch.object(
                LocalModelSmokeTester, "run_verification", return_value=mock_summary
            ) as mock_run:
                exit_code = LocalModelSmokeTester.main(args)
                assert exit_code == 0
                mock_run.assert_called_once()

    def test_is_compressed_tensors_evaluates_dict_and_object(self) -> None:
        """Verify _is_compressed_tensors identifies compressed-tensors method."""
        tester = LocalModelSmokeTester()
        assert tester._is_compressed_tensors({"quant_method": "compressed-tensors"}) is True
        assert tester._is_compressed_tensors({"quant_method": "bitsandbytes"}) is False
        mock_obj = MagicMock()
        mock_obj.quant_method = "compressed-tensors"
        assert tester._is_compressed_tensors(mock_obj) is True

    def test_create_dequantize_config_handles_dict_to_dict_and_generic(self) -> None:
        """Verify _create_dequantize_config sets dequantize=True across input types."""
        tester = LocalModelSmokeTester()
        cfg_dict = {"format": "int4"}
        res1 = tester._create_dequantize_config(cfg_dict)
        assert getattr(res1, "dequantize", False) is True

        mock_obj = MagicMock()
        mock_obj.to_dict.return_value = {"format": "int4"}
        res2 = tester._create_dequantize_config(mock_obj)
        assert getattr(res2, "dequantize", False) is True

        res3 = tester._create_dequantize_config(object())
        assert getattr(res3, "dequantize", False) is True

    def test_build_quantization_config_returns_config_when_compressed(self) -> None:
        """Verify _build_quantization_config returns dequantize config for compressed models."""
        tester = LocalModelSmokeTester()
        mock_cfg = MagicMock()
        mock_cfg.quantization_config = {"quant_method": "compressed-tensors"}
        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg):
            result = tester._build_quantization_config(self._MOCK_RESOLVED_PATH)
            assert result is not None
            assert getattr(result, "dequantize", False) is True

    def test_build_quantization_config_returns_none_on_missing_or_error(self) -> None:
        """Verify _build_quantization_config returns None when not compressed or error occurs."""
        tester = LocalModelSmokeTester()
        mock_cfg = MagicMock()
        mock_cfg.quantization_config = None
        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_cfg):
            assert tester._build_quantization_config(self._MOCK_RESOLVED_PATH) is None

        with patch("transformers.AutoConfig.from_pretrained", side_effect=RuntimeError("load err")):
            assert tester._build_quantization_config(self._MOCK_RESOLVED_PATH) is None

    def test_load_model_and_tokenizer_invokes_hf_loaders(self) -> None:
        """Verify _load_model_and_tokenizer calls AutoModel and AutoTokenizer loaders."""
        tester = LocalModelSmokeTester()
        with (
            patch.object(tester, "_build_quantization_config", return_value=None),
            patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_load,
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_load,
        ):
            model, tok = tester._load_model_and_tokenizer(self._MOCK_RESOLVED_PATH)
            assert model == mock_model_load.return_value
            assert tok == mock_tok_load.return_value
            mock_model_load.assert_called_once()
            mock_tok_load.assert_called_once()
