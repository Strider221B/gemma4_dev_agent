# Plan: Gemma 4 Special Tokens Update

## Objective
Update the codebase to support the new Gemma 4 special tokens, resolving the `ValueError: Missing required special token: <start_of_turn>` encountered during Supervised Fine-Tuning (SFT).

## Steps

1. **Update `sft_trainer.py`**:
   - Open `src/training/sft_trainer.py`.
   - Locate the `_REQUIRED_SPECIAL_TOKENS` tuple within the `SFTTrainerPipeline` class.
   - Replace the existing legacy tokens with the new tokens extracted from the Gemma 4 chat template:
     - `<|turn>`
     - `<turn|>`
     - `<|tool>`
     - `<tool|>`
     - `<|tool_call>`
     - `<tool_call|>`
     - `<|tool_response>`
     - `<tool_response|>`
     - `<|think|>`
     - `<|channel>`
     - `<channel|>`
     - `<|image|>`
     - `<|audio|>`
     - `<|video|>`

2. **Audit Data Preparation Scripts**:
   - Review files responsible for trajectory synthesis and formatting, such as `src/data/chat_formatter.py` and `src/data/trajectory_synthesiser.py`.
   - Find and replace any explicit occurrences of old tokens (`<start_of_turn>`, `<end_of_turn>`, `<|thought|>`) with the appropriate new tags (`<|turn>`, `<turn|>`, `<|channel>thought`, etc.).

3. **Verify Functionality**:
   - Instantiate the `SFTTrainerPipeline` locally with a sample Gemma 4 tokenizer.
   - Ensure `_verify_special_tokens` completes without throwing `ValueError`.
   - Run local CI checks (`ruff`, `mypy`, `pytest`) to guarantee code correctness.
