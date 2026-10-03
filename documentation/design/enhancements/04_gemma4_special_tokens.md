# Design: Gemma 4 Special Tokens Update

## Context
The project was previously configured for Gemma 2/3 chat templates which utilized tags like `<start_of_turn>`, `<end_of_turn>`, `<|thought|>`, etc.
However, with the migration to Gemma 4, the model's tokenizer and chat template have transitioned to a new set of special tokens (e.g., `<|turn>`, `<turn|>`, `<|tool_call>`, `<tool_call|>`, `<|think|>`).
Running the SFT pipeline with the current configuration results in a `ValueError` during token verification because the Gemma 4 tokenizer lacks the old tokens.

## Proposed Changes

1. **Update Required Special Tokens in `SFTTrainerPipeline`**:
   The `_REQUIRED_SPECIAL_TOKENS` tuple in `src/training/sft_trainer.py` must be updated to reflect the new Gemma 4 token definitions derived from the provided chat template:
   - Control Tokens: `<|turn>`, `<turn|>`
   - Tool Tokens: `<|tool>`, `<tool|>`, `<|tool_call>`, `<tool_call|>`, `<|tool_response>`, `<tool_response|>`
   - Reasoning Tokens: `<|think|>`, `<|channel>`, `<channel|>`
   - Media Tokens: `<|image|>`, `<|audio|>`, `<|video|>`

2. **Verify and Update Data Preprocessing Pipelines**:
   The components that generate conversational trajectories (like `src/data/chat_formatter.py` and `src/data/trajectory_synthesiser.py`) must be audited. Any hardcoded references to legacy tokens (like `<start_of_turn>`) must be replaced with the new Gemma 4 tags or adapted to rely strictly on the Hugging Face `apply_chat_template` functionality using message dictionaries.
