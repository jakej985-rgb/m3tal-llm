# M3TAL inference server

`m3tal-lm serve` exposes a trained checkpoint over HTTP so M3TAL
components (router, plugins) can route language-model requests without
coupling to model internals. CPU-only deployment is the default; CUDA is
available via the `cuda` build tag (`docs/CUDA.md`).

## Endpoints

| Method | Path           | Purpose                                        |
|--------|----------------|------------------------------------------------|
| GET    | `/healthz`     | Liveness + parameter count                      |
| POST   | `/v1/generate` | Raw prompt completion                           |
| POST   | `/v1/chat`     | Chat completion (chatml.v1); `stream:true` → SSE |

### POST /v1/generate

```json
{"prompt": "Once upon", "max_tokens": 64, "temperature": 0.8,
 "top_k": 0, "top_p": 0.0, "seed": 19, "stop": ["<|im_end|>"]}
```

→ `{"text": "...", "usage": {...}}`

`temperature: 0` selects greedy argmax. `top_k`/`top_p` are optional
filters (0 = disabled). `seed` makes sampling deterministic.

### POST /v1/chat

```json
{"messages": [{"role":"user","content":"Hello"}],
 "max_tokens": 64, "temperature": 0.0, "stream": false}
```

→ `{"text": "...", "template": "m3tal.chatml.v1", "usage": {...}}`

With `"stream": true` the response is `text/event-stream`: each
`data:` line carries `{"delta": "..."}`, the final event carries
`{"usage": {...}}`, and the stream ends with `data: [DONE]`. Generation always appends the chat stop markers, so
`<|im_end|>` terminates output and is stripped.

### Errors

All errors return `{"error": "..."}` with an appropriate status:
`400` malformed JSON/invalid request, `413` prompt or body over the
safety limits, `503` server at capacity or queue wait timeout (with
`Retry-After: 1`), `504` request deadline or generation timeout,
`500` internal.

## Safety policy (configurable, not in weights)

`server.Policy` bounds every request independently of the checkpoint:

- `MaxBodyBytes` — request body cap (default 1 MiB)
- `MaxPromptTokens` — encoded prompt cap (default 4096)
- `MaxOutputTokens` — per-request generation cap (default 512)
- `RequestTimeout` — per-request deadline (default 2m)
- `MaxConcurrent` — generations running at once (default 4)
- `MaxQueue` — requests waiting for a slot (default 8; a negative
  value rejects excess requests immediately instead of queueing)
- `QueueTimeout` — maximum queue wait (default 10s)

CLI flags: `--max-prompt-tokens`, `--max-output-tokens`,
`--max-body-mb`, `--timeout`, `--max-concurrent`, `--max-queue`,
`--queue-timeout`. The safety layer lives in `internal/server` and can
be tuned or replaced without touching model weights.

## Scheduling and telemetry

Generation runs behind a slot semaphore: at most `MaxConcurrent`
requests decode at once, up to `MaxQueue` more wait for a slot, and a
waiter that exceeds `QueueTimeout` (or whose client disconnects) is
removed without touching the model. Admission is deterministic:

- no slot and a full queue or queueing disabled → `503` "server at capacity";
- queue wait exceeded → `503` “queue wait exceeded”;
- request deadline while queued → `504`.

Each generation is one KV-cache owner: the cache is created per request
and freed when the request returns, so concurrent requests never share
cache state.

Every successful response carries per-request telemetry:

```json
"usage": {
  "selected_backend": "cuda", "backend": "cpu",
  "prompt_tokens": 18, "output_tokens": 64,
  "fallback_count": 0, "transfer_count": 12, "transfer_bytes": 4096,
  "queue_ms": 0, "generation_ms": 832
}
```

`selected_backend` is what kernel selection asked for; `backend` is what
actually ran, so a CUDA build that fell back to CPU is visible.
`fallback_count` counts tokens that needed a full forward pass because
the KV cache reached the context limit. Transfer counters are populated
by the CUDA backend and omitted on CPU. Non-streaming responses also
set `X-M3tal-Backend`, `X-M3tal-Queue-Ms` and `X-M3tal-Generation-Ms`
headers. `/healthz` reports `backend` (last backend used, or `unloaded`),
`max_concurrent`, `max_queue`, `inflight` and `queued`.

Race-enabled tests cover parallel requests, overload, queue timeout,
cancellation and connection shutdown:

```sh
go test -race ./internal/server ./internal/nn
```

## Go client

`internal/llmclient` wraps the HTTP API and includes a `Generator`
adapter whose signature matches the `m3tal-ai` assistant `Generator`
interface (`Generate(prompt, maxTokens, temperature, seed)`) — the
assistant lives in `m3tal-plugins/m3tal-ai` — so plugins can route
requests to a local server without this repo depending on it:

```go
c := llmclient.New("http://127.0.0.1:8080")
gen := llmclient.NewGenerator(c)
// In m3tal-plugins/m3tal-ai: assistant.SetLanguageModel(gen)
```

`Client.ChatStream` supports SSE consumption; an emit error cancels the
stream cleanly. Server unavailability surfaces as ordinary Go errors —
callers should treat the model as optional and fall back to deterministic
routing (the plugin already does this when no language model is loaded).

## Run

```sh
go run ./cmd/m3tal-lm serve --checkpoint models/stage0-tiny-stories.json \
  --addr 127.0.0.1:8080
```

Integration tests: `internal/server/server_test.go` (endpoints, SSE,
validation, limits) and `internal/llmclient/client_test.go` (client,
streaming, unavailability).
