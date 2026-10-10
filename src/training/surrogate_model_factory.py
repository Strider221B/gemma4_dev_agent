"""Factory for generating minimal surrogate models for local testing."""

from __future__ import annotations

import os
from typing import Any


class SurrogateModelFactory:
    """Creates lightweight causal language models and tokenizers for local smoke testing."""

    _VOCAB_SIZE: int = 1000
    _HIDDEN_SIZE: int = 64
    _INTERMEDIATE_SIZE: int = 128
    _NUM_HIDDEN_LAYERS: int = 2
    _NUM_ATTENTION_HEADS: int = 2
    _MAX_POSITION_EMBEDDINGS: int = 512
    _TARGET_MODULES: tuple[str, ...] = ("q_proj", "v_proj")
    _GEMMA4_SPECIAL_TOKENS: tuple[str, ...] = (
        "<|turn>",
        "<turn|>",
        "<|tool>",
        "<tool|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|think|>",
        "<|channel>",
        "<channel|>",
        "<|image|>",
        "<|audio|>",
        "<|video|>",
    )
    _TOKEN_UNK: str = "<unk>"
    _TOKEN_BOS: str = "<s>"
    _TOKEN_EOS: str = "</s>"
    _TOKEN_PAD: str = "<pad>"
    _TOKEN_PREFIX: str = "<token_"
    _TOKEN_SUFFIX: str = ">"
    _ADDITIONAL_SPECIAL_TOKENS_KEY: str = "additional_special_tokens"
    _CHAT_TEMPLATE: str = (
        "{% for message in messages %}"
        "{{ message['role'] + ': ' + message['content'] + '\n' }}"
        "{% endfor %}"
    )
    _UNK_TOKEN_ID: int = 0
    _BOS_TOKEN_ID: int = 1
    _EOS_TOKEN_ID: int = 2
    _PAD_TOKEN_ID: int = 3

    def __init__(self) -> None:
        """Initialize SurrogateModelFactory."""

    def create_surrogate_model(self, output_dir: str) -> str:
        """Generate and save minimal surrogate model and tokenizer to disk.

        Args:
            output_dir: Destination directory path for model artifacts.

        Returns:
            Output directory path string containing persisted artifacts.
        """
        os.makedirs(output_dir, exist_ok=True)
        config = self._build_model_config()
        self._save_model(config, output_dir)
        self._save_tokenizer(output_dir)
        return output_dir

    def get_target_modules(self) -> list[str]:
        """Return list of supported target projection layer names.

        Returns:
            List of module names targeted by LoRA adapter.
        """
        return list(self._TARGET_MODULES)

    def _build_model_config(self) -> object:
        """Build minimal causal language model configuration."""
        from transformers import LlamaConfig

        return LlamaConfig(
            vocab_size=self._VOCAB_SIZE,
            hidden_size=self._HIDDEN_SIZE,
            intermediate_size=self._INTERMEDIATE_SIZE,
            num_hidden_layers=self._NUM_HIDDEN_LAYERS,
            num_attention_heads=self._NUM_ATTENTION_HEADS,
            max_position_embeddings=self._MAX_POSITION_EMBEDDINGS,
            bos_token_id=self._BOS_TOKEN_ID,
            eos_token_id=self._EOS_TOKEN_ID,
            pad_token_id=self._PAD_TOKEN_ID,
        )

    def _build_vocab(self) -> dict[str, int]:
        """Build vocabulary mapping containing required special tokens up to vocab size."""
        vocab: dict[str, int] = {
            self._TOKEN_UNK: self._UNK_TOKEN_ID,
            self._TOKEN_BOS: self._BOS_TOKEN_ID,
            self._TOKEN_EOS: self._EOS_TOKEN_ID,
            self._TOKEN_PAD: self._PAD_TOKEN_ID,
        }
        for token in self._GEMMA4_SPECIAL_TOKENS:
            vocab[token] = len(vocab)
        for token_idx in range(len(vocab), self._VOCAB_SIZE):
            vocab[f"{self._TOKEN_PREFIX}{token_idx}{self._TOKEN_SUFFIX}"] = token_idx
        return vocab

    def _save_model(self, config: object, output_dir: str) -> None:
        """Instantiate and persist causal language model to disk."""
        from transformers import LlamaForCausalLM

        model_cls: Any = LlamaForCausalLM
        model: Any = model_cls(config)
        model.save_pretrained(output_dir)

    def _save_tokenizer(self, output_dir: str) -> None:
        """Build and save tokenizer containing required special tokens."""
        from tokenizers import Tokenizer as RawTokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast

        vocab = self._build_vocab()
        wl_cls: Any = WordLevel
        tok_cls: Any = RawTokenizer
        fast_tok_cls: Any = PreTrainedTokenizerFast
        raw_tok = tok_cls(wl_cls(vocab, unk_token=self._TOKEN_UNK))
        tokenizer = fast_tok_cls(
            tokenizer_object=raw_tok,
            bos_token=self._TOKEN_BOS,
            eos_token=self._TOKEN_EOS,
            unk_token=self._TOKEN_UNK,
            pad_token=self._TOKEN_PAD,
            chat_template=self._CHAT_TEMPLATE,
        )
        tokenizer.add_special_tokens(
            {self._ADDITIONAL_SPECIAL_TOKENS_KEY: list(self._GEMMA4_SPECIAL_TOKENS)}
        )
        tokenizer.save_pretrained(output_dir)
