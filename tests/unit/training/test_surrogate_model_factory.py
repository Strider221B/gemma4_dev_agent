"""Unit tests for SurrogateModelFactory."""

from __future__ import annotations

import os
import tempfile
from typing import Any, ClassVar

from transformers import AutoTokenizer

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

    def test_instantiation_succeeds(self) -> None:
        """Verify SurrogateModelFactory instantiates successfully."""
        factory = SurrogateModelFactory()
        assert isinstance(factory, SurrogateModelFactory)

    def test_get_target_modules_returns_expected_layers(self) -> None:
        """Verify get_target_modules returns expected projection layer names."""
        factory = SurrogateModelFactory()
        assert factory.get_target_modules() == self._EXPECTED_TARGETS

    def test_build_model_config_has_expected_dimensions(self) -> None:
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

    def test_create_surrogate_model_persists_artifacts(self) -> None:
        """Verify create_surrogate_model generates and saves files to target directory."""
        factory = SurrogateModelFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = factory.create_surrogate_model(tmpdir)
            assert result == tmpdir
            assert os.path.isfile(os.path.join(tmpdir, self._CONFIG_FILE))
            assert os.path.isfile(os.path.join(tmpdir, self._TOKENIZER_CONFIG_FILE))

    def test_create_surrogate_model_tokenizer_loadable_with_special_tokens(self) -> None:
        """Verify persisted tokenizer can be loaded with AutoTokenizer and has special tokens."""
        factory = SurrogateModelFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            factory.create_surrogate_model(tmpdir)
            tokenizer = AutoTokenizer.from_pretrained(tmpdir)
            vocab = tokenizer.get_vocab()
            for token in factory._GEMMA4_SPECIAL_TOKENS:
                assert token in vocab
            assert tokenizer.chat_template is not None
