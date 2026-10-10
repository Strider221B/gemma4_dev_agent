"""Unit tests for SurrogateModelFactory."""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from src.training.surrogate_model_factory import SurrogateModelFactory


class TestSurrogateModelFactory:
    """Test suite verifying SurrogateModelFactory configuration and artifact creation."""

    _EXPECTED_TARGETS: ClassVar[list[str]] = ["q_proj", "v_proj"]
    _EXPECTED_VOCAB_SIZE: ClassVar[int] = 1000
    _EXPECTED_HIDDEN_SIZE: ClassVar[int] = 64
    _EXPECTED_INTERMEDIATE_SIZE: ClassVar[int] = 128
    _EXPECTED_NUM_LAYERS: ClassVar[int] = 2
    _EXPECTED_NUM_HEADS: ClassVar[int] = 2
    _EXPECTED_MAX_POS: ClassVar[int] = 512
    _CONFIG_FILE: ClassVar[str] = "config.json"
    _TOKENIZER_CONFIG_FILE: ClassVar[str] = "tokenizer_config.json"
    _KEY_TRANSFORMERS: ClassVar[str] = "transformers"
    _KEY_TOKENIZERS: ClassVar[str] = "tokenizers"
    _KEY_TOKENIZERS_MODELS: ClassVar[str] = "tokenizers.models"
    _DUMMY_CHAT_TEMPLATE: ClassVar[str] = "dummy_chat_template"
    _EMPTY_JSON: ClassVar[str] = "{}"
    _UTF8: ClassVar[str] = "utf-8"

    @pytest.fixture
    def mock_env(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
        """Set up mocked transformers and tokenizers modules in sys.modules."""
        mocks = self._build_mocks()
        monkeypatch.setitem(sys.modules, self._KEY_TRANSFORMERS, mocks[self._KEY_TRANSFORMERS])
        monkeypatch.setitem(sys.modules, self._KEY_TOKENIZERS, mocks[self._KEY_TOKENIZERS])
        monkeypatch.setitem(
            sys.modules, self._KEY_TOKENIZERS_MODELS, mocks[self._KEY_TOKENIZERS_MODELS]
        )
        return mocks

    def test_instantiation_succeeds(self) -> None:
        """Verify SurrogateModelFactory instantiates successfully."""
        factory = SurrogateModelFactory()
        assert isinstance(factory, SurrogateModelFactory)

    def test_get_target_modules_returns_expected_layers(self) -> None:
        """Verify get_target_modules returns expected projection layer names."""
        factory = SurrogateModelFactory()
        assert factory.get_target_modules() == self._EXPECTED_TARGETS

    def test_build_model_config_has_expected_dimensions(
        self, mock_env: dict[str, MagicMock]
    ) -> None:
        """Verify _build_model_config produces LlamaConfig with miniature dimensions."""
        factory = SurrogateModelFactory()
        cfg: Any = factory._build_model_config()
        assert cfg.vocab_size == self._EXPECTED_VOCAB_SIZE
        assert cfg.hidden_size == self._EXPECTED_HIDDEN_SIZE
        assert cfg.intermediate_size == self._EXPECTED_INTERMEDIATE_SIZE
        assert cfg.num_hidden_layers == self._EXPECTED_NUM_LAYERS
        assert cfg.num_attention_heads == self._EXPECTED_NUM_HEADS
        assert cfg.max_position_embeddings == self._EXPECTED_MAX_POS

    def test_build_vocab_contains_all_special_tokens(self) -> None:
        """Verify _build_vocab includes all 14 Gemma 4 special tokens and fills vocab."""
        factory = SurrogateModelFactory()
        vocab = factory._build_vocab()
        assert len(vocab) == self._EXPECTED_VOCAB_SIZE
        for token in factory._GEMMA4_SPECIAL_TOKENS:
            assert token in vocab

    def test_create_surrogate_model_persists_artifacts(
        self, mock_env: dict[str, MagicMock]
    ) -> None:
        """Verify create_surrogate_model generates and saves files to target directory."""
        factory = SurrogateModelFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = factory.create_surrogate_model(tmpdir)
            assert result == tmpdir
            assert os.path.isfile(os.path.join(tmpdir, self._CONFIG_FILE))
            assert os.path.isfile(os.path.join(tmpdir, self._TOKENIZER_CONFIG_FILE))

    def test_create_surrogate_model_tokenizer_configured(
        self, mock_env: dict[str, MagicMock]
    ) -> None:
        """Verify create_surrogate_model adds special tokens to tokenizer."""
        factory = SurrogateModelFactory()
        mock_tf = mock_env[self._KEY_TRANSFORMERS]
        with tempfile.TemporaryDirectory() as tmpdir:
            factory.create_surrogate_model(tmpdir)
            mock_tok = mock_tf.PreTrainedTokenizerFast.return_value
            mock_tok.add_special_tokens.assert_called_once()
            mock_tok.save_pretrained.assert_called_once_with(tmpdir)

    def test_create_surrogate_model_tokenizer_loadable(
        self, mock_env: dict[str, MagicMock]
    ) -> None:
        """Verify persisted tokenizer can be loaded with AutoTokenizer."""
        factory = SurrogateModelFactory()
        mock_tf = mock_env[self._KEY_TRANSFORMERS]
        mock_loaded_tok = MagicMock()
        mock_loaded_tok.get_vocab.return_value = {
            token: idx for idx, token in enumerate(factory._GEMMA4_SPECIAL_TOKENS)
        }
        mock_loaded_tok.chat_template = self._DUMMY_CHAT_TEMPLATE
        mock_tf.AutoTokenizer.from_pretrained.return_value = mock_loaded_tok
        with tempfile.TemporaryDirectory() as tmpdir:
            factory.create_surrogate_model(tmpdir)
            loaded = mock_tf.AutoTokenizer.from_pretrained(tmpdir)
            mock_tf.AutoTokenizer.from_pretrained.assert_called_with(tmpdir)
            vocab = loaded.get_vocab()
            for token in factory._GEMMA4_SPECIAL_TOKENS:
                assert token in vocab
            assert loaded.chat_template is not None

    def _build_mocks(self) -> dict[str, MagicMock]:
        """Construct dictionary of mocks for external dependencies."""
        mock_tf = self._create_mock_transformers()
        mock_tokenizers = MagicMock()
        mock_tokenizers_models = MagicMock()
        return {
            self._KEY_TRANSFORMERS: mock_tf,
            self._KEY_TOKENIZERS: mock_tokenizers,
            self._KEY_TOKENIZERS_MODELS: mock_tokenizers_models,
        }

    def _create_mock_transformers(self) -> MagicMock:
        """Create mock transformers module with configurable classes."""
        mock_tf = MagicMock()

        def _mock_llama_config(**kwargs: Any) -> MagicMock:
            cfg = MagicMock()
            for key, val in kwargs.items():
                setattr(cfg, key, val)
            return cfg

        mock_tf.LlamaConfig = MagicMock(side_effect=_mock_llama_config)
        mock_model = MagicMock()
        mock_model.save_pretrained.side_effect = self._write_config_file
        mock_tf.LlamaForCausalLM = MagicMock(return_value=mock_model)
        mock_tokenizer = MagicMock()
        mock_tokenizer.save_pretrained.side_effect = self._write_tokenizer_config_file
        mock_tf.PreTrainedTokenizerFast = MagicMock(return_value=mock_tokenizer)
        return mock_tf

    def _write_config_file(self, target_dir: str) -> None:
        """Write placeholder config.json simulating model.save_pretrained."""
        file_path = os.path.join(target_dir, self._CONFIG_FILE)
        with open(file_path, "w", encoding=self._UTF8) as file_handle:
            file_handle.write(self._EMPTY_JSON)

    def _write_tokenizer_config_file(self, target_dir: str) -> None:
        """Write placeholder tokenizer_config.json simulating save_pretrained."""
        file_path = os.path.join(target_dir, self._TOKENIZER_CONFIG_FILE)
        with open(file_path, "w", encoding=self._UTF8) as file_handle:
            file_handle.write(self._EMPTY_JSON)
