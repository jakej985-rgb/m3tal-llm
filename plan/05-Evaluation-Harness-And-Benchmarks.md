# 05 Evaluation Harness and Benchmarks

## Goal
Establish repeatable, automated benchmark suites to ensure the model does not regress, verifies genuine linguistic fluency, and guarantees 100% valid tool planning.

## Test Matrix

### 1. Quantitative Benchmark (`m3tal-lm eval-bench`)
- **Held-Out Perplexity**: Evaluated on `data/curriculum-val.txt` using `--eval-mode fast`.
- **Target**: Perplexity **< 15.0** (Cross-entropy loss < 2.70 nats).
- **Regression Guard**: Fails CI build if loss increases by more than 0.05 nats vs. baseline report (`--max-loss-increase 0.05`).

### 2. Conversational Evaluation Suite (`m3tal-lm chat-eval`)
Evaluates against held-out multi-turn dialogues (`data/chat-eval-suite.jsonl`) across 4 categories:
1. **Direct Questions**: Answers clearly without rambling or hallucinating missing sources.
2. **Multi-Turn Context**: Remembers variables/targets introduced in previous turns (*"My name is Sam"*, *"restart web"*, *"what did I ask?"*).
3. **Format Discipline**: Adheres to strict one-word, JSON, or bullet constraints when instructed.
4. **Tool Planning Syntactic Validity**: 100% of generated plans must parse into valid JSON matching the `{"steps":[{"tool":"...", "target":"..."}]}` schema.

### 3. Generation Quality & Repetition Check
- **N-gram Repetition Score**: Verifies that 3-gram and 4-gram loop degeneration is below 1% under `--top-p 0.9 --temperature 0.7`.
- **Stop Marker Adherence**: Asserts that generations terminate on `<|im_end|>` and do not leak special tokens into user-visible buffers.

## Steps
1. Create benchmark report target directory `reports/eval-bench/`.
2. Generate baseline report for `models/m3tal-smart-6m.json`.
3. Add an automated evaluation script `scripts/eval-model.sh` that runs:
   - `eval-bench`
   - `chat-eval`
   - Plan schema validation test
4. Log metrics into `BASELINE.md`.

## Definition of Done
- `m3tal-lm eval-bench` executes with zero errors.
- Schema validator confirms 100% of tool plans parse into `ToolStep` structures.
- JSON report committed to `reports/eval-bench/`.
