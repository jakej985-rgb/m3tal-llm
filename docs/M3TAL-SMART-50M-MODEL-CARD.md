# Model Card: M3TAL Smart 50M (Code-Smart)

## Summary

- Checkpoint: `models/m3tal-smart-50m.json`
- Parameters: 43.61M
- Vocab size: 1536 (tokenizer fnv64a:770ba10b18c9649b)
- Context: 512, width: 512, layers: 16, heads: 16
- Training step: 550
- Chat template: `m3tal.chatml.v1`

## Intended use

Local, CPU-first conversational language model experimentation.
Intended for research, evaluation, and M3TAL integration testing.
Not intended for production decision-making, medical/legal advice,
or deployment to untrusted multi-user environments without further
safety review.

## Limitations

- Small model trained from scratch; output quality is limited and may be incoherent, biased, or incorrect.
- May reproduce memorized training text; verify outputs before use.
- No built-in content moderation — apply safety policy at the serving layer (see docs/SERVE.md).
- CPU inference is slow for large checkpoints; check docs/PACKAGING.md hardware requirements.

## Training data and provenance

High-density curriculum combining 2,000 narrative stories, 841 ecosystem documentation sections, and 6,109 m3tal-codex / CoT instruction pairs covering Go, Docker, Compose, and Traefik.

## Evaluation

- No held-out conversational evaluation recorded.
- Automated metrics are heuristics; output quality requires human review.

## Hardware and runtime

CPU-first pure-Go inference; optional CUDA backend behind the `cuda`
build tag. See docs/CUDA.md and docs/PACKAGING.md.
