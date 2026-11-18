# 02 Tokenizer and Vocabulary Optimization

## Goal
Design and compile an optimized Byte-Pair Encoding (BPE) vocabulary (1,024 to 2,048 tokens) that eliminates character-by-character spelling breakdowns for technical terms while maximizing context window efficiency.

## Problem Statement
When technical words like `M3TAL`, `container`, `Debian`, or `dashboard` are missing from the tokenizer, they decompose into 6–10 raw byte tokens. In a 256-token context window, letter-by-letter spelling exhausts memory in under 25 words and forces the neural network into character-level hallucinations (e.g. `debiana`, `Sotupged`).

## Architectural Specifications

1. **Vocabulary Size**: 1,536 or 2,048 tokens
   - IDs 0..255: Raw UTF-8 bytes (ensures 100% vocabulary coverage without `<unk>`).
   - IDs 256..259: Special tokens (`<|endoftext|>`, `<|pad|>`, `<|im_start|>`, `<|im_end|>`).
   - IDs 260..2047: Learned merges prioritized by frequency in the curriculum.

2. **Semantic Density Targets**:
   - **M3TAL Core Entities**: `M3TAL`, `Core`, `Hub`, `plugin`, `supervisor`, `allowlist`, `telemetry`
   - **Infrastructure Terms**: `container`, `service`, `stack`, `Docker`, `Debian`, `Traefik`, `health`, `stats`, `logs`
   - **Actions & Verbs**: `restart`, `start`, `stop`, `confirm`, `inspect`, `deploy`, `status`
   - **Syntax & Structural Tokens**: `Thought:`, `PLAN:`, `{"steps":`, `{"tool":`, `"target":`

## Steps
1. Feed the combined curriculum dataset into `lm.TrainTokenizerWithOptions`.
2. Inspect the learned merge table (`tokens`) to verify that all primary technical entities are atomic tokens.
3. Measure token compression ratio:
   - Target: Average English word length compressed from 4.5 bytes to **1.3 tokens per word**.
4. Serialize the verified tokenizer artifact with SHA-256 fingerprinting.

## Definition of Done
- Tokenizer trained and verified with zero missing core keywords.
- Test suite verifies that `<|im_start|>` and `<|im_end|>` boundaries encode and decode without corrupting adjacent text.
- Tokenizer test pass in `internal/lm/tokenizer_test.go`.
