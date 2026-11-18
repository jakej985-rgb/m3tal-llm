# 01 Data Engine and Curriculum

## Goal
Build a high-density training corpus that teaches the model flawless English syntax, M3TAL ecosystem knowledge, and Chain-of-Thought reasoning without feeding noisy markdown dumps.

## Problem Statement
Feeding raw repository dumps containing YAML frontmatter, table pipes (`|---|`), and raw shell diffs causes catastrophic confusion in compact models (<10M params). The model must learn from structured, syntactically clean examples modeled after the TinyStories and Microsoft Phi curriculum methodologies.

## Corpus Pillars

1. **Syntactic Grammar Foundation (~150k tokens)**:
   - High-quality narrative prose from `data/m3tal-tales/m3tal_tales_tiny.jsonl` and `m3tal_tales_small.jsonl`.
   - Teaches the transformer core linguistic transitions, pronoun tracking (*"it"*, *"they"*), and coherent sentence closures.

2. **M3TAL Specification Knowledge (~200k tokens)**:
   - Extracted architectural and operational summaries from:
     - `M3tal-core` (Dashboard, System Monitor, telemetry, settings, allowlists)
     - `M3tal-Hub` (App catalog, releases, pages)
     - `m3tal-plugins` (Plugin supervisor, official plugin specs)
   - Formatted into clean explanatory prose paragraphs rather than fragmented tables.

3. **Conversational & Chain-of-Thought Q&A Pairs (~100k tokens)**:
   - Expanded dataset based on `data/chat-custom.jsonl` and `data/chat-instructions.jsonl`.
   - Every complex technical prompt includes an internal `Thought:` step:
     ```json
     {
       "messages": [
         {"role": "user", "content": "How do I check system performance and running containers?"},
         {"role": "assistant", "content": "Thought: The user is requesting system performance metrics and container inventory. These correspond to the stats and containers tools.\nTo check system performance and view active containers, run the stats and containers commands through Core."}
       ],
       "category": "ops-qa"
     }
     ```

## Steps
1. Create a corpus generation utility (`cmd/build-curriculum`) that cleans and converts raw ecosystem markdown into high-density declarative summaries.
2. Filter out raw code artifacts, broken links, base64 strings, and raw pipe tables.
3. Generate 300+ Chain-of-Thought (CoT) Q&A pairs covering:
   - Diagnostics and health checks
   - Container management and service lifecycle
   - Plugin supervisor and allowlist confirmation semantics
   - Architecture overviews and configuration
4. Build deterministic 90% train / 10% validation held-out splits with provenance tracking (`docs/DATA.md`).

## Definition of Done
- Clean curriculum file `data/curriculum-train.txt` and `data/curriculum-val.txt` generated.
- Q&A dataset `data/chat-curriculum.jsonl` validated with zero format errors via `m3tal-lm chatdata validate`.
- Provenance manifest generated in `data/provenance/curriculum.sources.json`.
