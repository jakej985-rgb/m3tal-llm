# M3TAL Pure-Go LLM: High-Efficiency Intelligent Assistant Plan

## Vision
Transform `m3tal-llm` into a compact, highly efficient language model written **entirely in pure Go** that runs offline on local CPU (<50ms latency, <40MB RAM) and outperforms conventional rule-based chatbots through genuine semantic understanding, structured reasoning, and native tool planning.

## Phases Index

| Phase | Document | Description |
| :--- | :--- | :--- |
| **01** | [`01-Data-Engine-And-Curriculum.md`](01-Data-Engine-And-Curriculum.md) | High-density synthetic dataset engine & curriculum design (grammar + specs + CoT). |
| **02** | [`02-Tokenizer-And-Vocabulary-Optimization.md`](02-Tokenizer-And-Vocabulary-Optimization.md) | Calibrated BPE vocabulary (1,024–2,048 tokens) eliminating character-spelling bottlenecks. |
| **03** | [`03-Model-Architecture-Scaling.md`](03-Model-Architecture-Scaling.md) | Scaling the pure-Go Transformer from 0.9M to ~6.5M parameters with RoPE, RMSNorm, and SwiGLU. |
| **04** | [`04-Pretraining-And-SFT-Pipeline.md`](04-Pretraining-And-SFT-Pipeline.md) | Two-stage pretraining foundation and Chain-of-Thought Supervised Fine-Tuning (SFT). |
| **05** | [`05-Evaluation-Harness-And-Benchmarks.md`](05-Evaluation-Harness-And-Benchmarks.md) | Automated regression test suites, perplexity tracking, and tool plan validation. |
| **06** | [`06-Ecosystem-Integration-And-Serving.md`](06-Ecosystem-Integration-And-Serving.md) | Fast local HTTP inference (`/v1/chat`), SSE streaming, and integration with `m3tal-ai` & Core. |
| **99** | [`99-Rollout-Order.md`](99-Rollout-Order.md) | Step-by-step rollout sequence, dependency graph, and definition of done. |

## Core Principles
1. **100% Pure Go & Self-Owned**: Zero external AI APIs, zero Python dependencies, zero cloud latency.
2. **Efficiency Over Brute Force**: Maximizing intelligence per parameter through clean curriculum data and Chain-of-Thought reasoning.
3. **Safety by Construction**: Model proposes structured plans; the M3TAL Core host validates targets against live allowlists and requires operator confirmation for mutations.
