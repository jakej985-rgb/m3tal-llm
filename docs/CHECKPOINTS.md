# Checkpoint integrity, durability, and evaluation identity

A checkpoint is the only durable record of a training run, and an evaluation
report is only meaningful if it says exactly what it measured. This document
covers both: what `m3tal-lm` checks before trusting a checkpoint, how it writes
one, and what identity every evaluation result carries.

## Integrity checks on load

`LoadCheckpoint` refuses state that cannot be trained or evaluated instead of
loading it and producing silent nonsense. Both containers (JSON and
safetensors) run the same checks:

- **Model shape** — `Config.Validate`: context 2..2048, width 2..2048,
  layers 1..16, heads 1..32, and `width % heads == 0`. A negative `layers` or
  `heads` is an error rather than a silent default (0 still means "use the
  default", for older checkpoints that omitted them).
- **Tokenizer state** — `Tokenizer.Validate`: vocabulary within
  `MinVocabSize..MaxVocabSize`, no empty or duplicate token strings, merges
  whose left/right/result are all real tokens, and special tokens pointing at
  existing, correctly named ids. A malformed tokenizer would mis-tokenize
  every later evaluation, so loading stops here.
- **Weights and optimizer moments** — every tensor and every Adam moment must
  be finite; a NaN or ±Inf is reported with its parameter name and index.
  Second moments must also be non-negative, since they are squared gradients.
- **Optimizer layout** — moment arrays must have exactly one entry per weight
  (counted for JSON, shape-checked per tensor for safetensors), and a
  checkpoint that declares `has_optimizer=1` must actually contain both
  moment tensors.
- **Step** — a negative `step` is rejected. A malformed `step` or `seed` value
  is an error instead of silently loading as 0.
- **Container version** — JSON `format_version` above 2 and safetensors
  `format_version` above 1 are rejected rather than partially understood.
- **Safetensors layout** — the tensor table is validated as a whole before any
  tensor is read: entries must be well formed, byte ranges must fit the data
  region, tensors may not overlap, F32 tensor byte spans must match their
  declared shape, the declared `parameter_count` must match the model, and the
  tensors must cover the data region *exactly*. That last check catches both
  truncation and trailing garbage.
- **Tokenizer fingerprint** — when the container records a
  `tokenizer_hash`, the deserialized tokenizer must still hash to it.

Saving applies the same model-state rules: `SaveCheckpoint`,
`SaveSafetensors` and `SaveSafetensorsCheckpoint` refuse to write non-finite
weights or moments, so a broken model is never persisted.

## Legacy checkpoints

Checkpoints written before the RoPE/SwiGLU rewrite do not match the current
parameter layout, so `LoadCheckpoint` rejects them. Two such layouts exist:

| Layout | Positional encoding | Norms | Feed-forward |
|---|---|---|---|
| `v1.pos_emb.mlp` | learned `[context,width]` table | none | `ffIn` → tanh → `ffOut` |
| `v1.rope.mlp` | RoPE | two gammas per block | `ffIn` → tanh → `ffOut` |

The rejection names the detected layout and the command that fixes it:

```console
$ m3tal-lm migrate --in models/m3tal-base.json --out models/m3tal-base.migrated.json
migrate models/m3tal-base.json
  layout v1.pos_emb.mlp: 49984 -> 46144 parameters; dropped 8192 learned positional values (RoPE now); RMSNorm gammas initialized to 1; tanh MLP mapped to SwiGLU (gate=ffIn, up=1, down=ffOut), so the activation differs until fine-tuning
```

`migrate` identifies the layout from the parameter count alone (never from the
file name), maps it onto the current architecture, writes a current checkpoint
with the atomic JSON writer, and reloads it to prove it loads:

- token embedding, attention projections and the output projection are copied
  value for value;
- a learned positional table is dropped, because the current architecture
  encodes position with RoPE inside each attention block;
- blocks without stored gammas keep the identity initialization (`gamma = 1`).
  Those legacy blocks had no normalization at all, so this is the closest
  available setting: the pre-norm RMSNorm still rescales each sublayer input to
  unit RMS, but adds no learnable scaling on top of it;
- the two-matrix MLP becomes the SwiGLU triple `gate = ffIn`, `up = 1` (zero
  weights, unit bias), `down = ffOut`. Both linear maps survive exactly, so
  only the activation changes: `silu` replaces `tanh`;
- optimizer moments are dropped and `step` restarts at 0, because they belong
  to the old graph.

A migrated checkpoint keeps its learned representations but is not numerically
identical to the original: the activation changed (`silu` for `tanh`), blocks
that had no normalization now run through RMSNorm, and a learned positional
table became RoPE. It is a fine-tuning starting point, not a drop-in
replacement. Score it with
`eval-bench` before relying on it. The importer is the last consumer of those
layouts; new checkpoints are always written in the current one, so a migrated
file can never drift back into a legacy shape.

### Known-broken runs (deliberately not re-exported)

Three legacy runs were unlearned before the architecture change, so migrating
them only reproduces a broken model. They are kept as originals for the record
but have no `.migrated.json`; do not re-create one unless the run is retrained.

| Checkpoint | Layout | Original-era loss | Migrated loss |
|---|---|---|---|
| `models/m3tal-instruct.json` | `v1.pos_emb.mlp` | 25.34 | 53.5 |
| `models/m3tal-instruct-v3.json` | `v1.pos_emb.mlp` | 13.53 | 24.2 |
| `models/m3tal-instruct-v4.json` | `v1.pos_emb.mlp` | 17.56 | 26.3 |

All figures are nats per token on the 25-line instruction corpus. "Original-era"
means each file scored by the code of its own commit (`734f874`) rather than by
the current loader, which is the fair comparison: these files were already far
above chance (`ln 384 = 5.95`) for their own code, so migration is not what broke
them. `models/m3tal-base.json` scores 3.71 that way (4.78 after migration), and
`m3tal-cot.json` and `m3tal-instruct-v2.json` (byte-identical twins) score 5.14
migrated, so the importer itself is sound.

## Atomic saves and durability

Both writers (`saveJSONCheckpoint`, `writeSafetensors`) go through
`writeFileAtomic`:

1. create a uniquely named temporary file (`.NAME.tmp-*`, mode 0600) in the
   destination directory, so concurrent saves cannot collide and a crash
   leaves no half-written file under the real name;
2. write the full container;
3. `fsync` the file;
4. `rename` it over the destination (atomic within a directory);
5. `fsync` the directory entry (best effort).

A reader therefore sees either the previous complete checkpoint or the new
complete one. On error or interruption the temporary file is removed and the
previous checkpoint is left untouched.

Durability expectations:

- On ext4/xfs with default mount options, a successful save survives a power
  loss: file contents are flushed before the rename and the rename is flushed
  after it.
- If the destination filesystem rejects directory `fsync` (some network and
  FUSE filesystems do), the call still succeeds but a crash immediately after
  the rename may lose the new name. The previous checkpoint is never
  corrupted either way.
- Renames are atomic only within one filesystem. Saving across a mount
  boundary with a symlinked destination is not atomic; write to a path on the
  same filesystem as the final location.
- `fsync` guarantees the data reached the device, not that the device honours
  its cache; use storage with a battery/flash-backed write cache when a lost
  checkpoint would be expensive.

## Evaluation identity

Every eval report now carries an `identity` object:

| Field | Meaning |
|---|---|
| `evaluator_version` | Scoring semantics version (`eval/1`). Bumped whenever loss computation, truncation or decoding defaults change. |
| `eval_mode` | `fast`, `exact`, or `chat`. |
| `backend` | `kernel.Selected()` at run time (`cpu`, `cuda`). |
| `checkpoint_sha256` | SHA-256 of the checkpoint file bytes. |
| `model_sha256` | SHA-256 over config + tokenizer fingerprint + every weight bit, so the same weights loaded from either container share one identity. |
| `tokenizer_hash` | `TokenizerFingerprint` (fnv64a over vocabulary and merges). |
| `model_config`, `context` | Shape and context window actually used. |
| `dataset_sha256` | SHA-256 of the corpus files in order, or of the in-memory texts (chat suites) when no path is used. |
| `dataset_split`, `dataset_items`, `dataset_tokens` | Split name, item count, and the corpus token count before any `--max-eval-tokens` cap. |
| `eval_tokens` | Tokens actually scored. |

`eval-bench` and `chat-eval` both record it, and `--baseline` comparisons use
it:

- **Refused** (the command exits non-zero, no delta is reported) when the
  evaluator version, eval mode, context, model config, tokenizer, dataset
  fingerprint, or scored token count differs — a loss delta between those runs
  would not mean anything.
- **Warned** when only the backend differs (CPU and CUDA reduce in a different
  order, so tiny loss differences are expected), when the split name differs,
  when the baseline used the same checkpoint (the delta is noise), or when the
  baseline predates identity recording (the comparison is unverified).

## Reproducibility

Given the same checkpoint bytes, corpus bytes and `--eval-mode`, evaluation is
deterministic on the CPU backend: `Model.Evaluate` and `Model.EvaluateFast` are
pure functions of the loaded weights and the text. The tests assert that two
loads of one checkpoint produce identical fingerprints and identical loss, and
that JSON and safetensors containers of the same model produce the same
fingerprint. CUDA results are close but not bit-identical to CPU (different
reduction order), which is why the backend is recorded rather than assumed.
