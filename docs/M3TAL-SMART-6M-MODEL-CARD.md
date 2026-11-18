# Model Card: m3tal-smart-6m

## Summary

- Checkpoint: `models/m3tal-smart-6m.json`
- Parameters: 6.05M
- Vocab size: 1536 (tokenizer fnv64a:770ba10b18c9649b)
- Context: 256, width: 256, layers: 8, heads: 8
- Training step: 600
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

Pretrained on data/curriculum-train.txt (M3TAL lore, apps architecture, CoT pairs) and fine-tuned on data/chat-curriculum.jsonl (109 reasoning chat examples).

## Evaluation

- No held-out conversational evaluation recorded.
- Automated metrics are heuristics; output quality requires human review.

## Hardware and runtime

CPU-first pure-Go inference; optional CUDA backend behind the `cuda`
build tag. See docs/CUDA.md and docs/PACKAGING.md.
