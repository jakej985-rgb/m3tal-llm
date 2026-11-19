// Package server exposes a trained lm.Model over HTTP for M3TAL
// integration (issue: Stage 4 M3TAL router integration + safety
// controls). The safety policy is configuration, not model weights:
// request size, prompt/output token limits, and timeouts are enforced
// here, independently of the checkpoint.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"m3tal-llm/internal/lm"
)

// Policy bounds request cost, output size, and scheduling. Zero values mean
// "use DefaultPolicy"; a negative MaxQueue means "reject excess requests
// immediately instead of queueing them".
type Policy struct {
	MaxBodyBytes    int64         // request body cap (default 1 MiB)
	MaxPromptTokens int           // encoded prompt cap (default 4096)
	MaxOutputTokens int           // generation cap (default 512)
	RequestTimeout  time.Duration // per-request cap (default 2m)
	AllowedRoles    map[string]bool

	// MaxConcurrent bounds generations running at once.
	MaxConcurrent int // default 4
	// MaxQueue bounds requests waiting for a slot (default 8).
	MaxQueue int // negative = no waiting queue
	// QueueTimeout bounds how long a request waits for a slot.
	QueueTimeout time.Duration // default 10s
}

// DefaultPolicy is the built-in safety boundary.
var DefaultPolicy = Policy{
	MaxBodyBytes:    1 << 20,
	MaxPromptTokens: 4096,
	MaxOutputTokens: 512,
	RequestTimeout:  2 * time.Minute,
	MaxConcurrent:   4,
	MaxQueue:        8,
	QueueTimeout:    10 * time.Second,
}

func (p Policy) withDefaults() Policy {
	if p.MaxBodyBytes <= 0 {
		p.MaxBodyBytes = DefaultPolicy.MaxBodyBytes
	}
	if p.MaxPromptTokens <= 0 {
		p.MaxPromptTokens = DefaultPolicy.MaxPromptTokens
	}
	if p.MaxOutputTokens <= 0 {
		p.MaxOutputTokens = DefaultPolicy.MaxOutputTokens
	}
	if p.RequestTimeout <= 0 {
		p.RequestTimeout = DefaultPolicy.RequestTimeout
	}
	if p.MaxConcurrent <= 0 {
		p.MaxConcurrent = DefaultPolicy.MaxConcurrent
	}
	if p.MaxQueue == 0 {
		p.MaxQueue = DefaultPolicy.MaxQueue
	}
	if p.MaxQueue < 0 {
		p.MaxQueue = 0
	}
	if p.QueueTimeout <= 0 {
		p.QueueTimeout = DefaultPolicy.QueueTimeout
	}
	return p
}

// Server serves lm.Model inference endpoints with bounded concurrency:
// a generation slot semaphore plus a bounded, time-limited waiting queue.
type Server struct {
	model  *lm.Model
	policy Policy
	mux    *http.ServeMux

	// slots holds one token per in-flight generation. len(slots) is the
	// number of generations currently running.
	slots chan struct{}

	// mu guards queued and lastBackend.
	mu          sync.Mutex
	queued      int
	lastBackend string
}

// New creates an inference server around model.
func New(model *lm.Model, policy Policy) *Server {
	effective := policy.withDefaults()
	s := &Server{
		model:  model,
		policy: effective,
		mux:    http.NewServeMux(),
		slots:  make(chan struct{}, effective.MaxConcurrent),
	}
	s.mux.HandleFunc("GET /healthz", s.handleHealth)
	s.mux.HandleFunc("POST /v1/generate", s.handleGenerate)
	s.mux.HandleFunc("POST /v1/chat", s.handleChat)
	return s
}

// Inflight reports the number of generations currently holding a slot.
func (s *Server) Inflight() int { return len(s.slots) }

// Queued reports the number of requests waiting for a slot.
func (s *Server) Queued() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.queued
}

// Handler returns the HTTP handler.
func (s *Server) Handler() http.Handler { return s.mux }

// Model returns the underlying model (for metadata endpoints).
func (s *Server) Model() *lm.Model { return s.model }

// Policy returns the effective safety policy.
func (s *Server) Policy() Policy { return s.policy }

type errorResponse struct {
	Error string `json:"error"`
}

func writeError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(errorResponse{Error: msg})
}

// Usage is the per-request telemetry returned by the generation endpoints
// and emitted as the final SSE event of a stream. The embedded stats are
// flattened into the same JSON object.
type Usage struct {
	lm.GenerateStats
	QueueMs      int64 `json:"queue_ms"`
	GenerationMs int64 `json:"generation_ms"`
}

var (
	errOverloaded   = errors.New("server is at capacity")
	errQueueTimeout = errors.New("queue wait timed out")
)

// acquire reserves a generation slot, waiting in the bounded queue if the
// server is busy. It returns how long the request waited. The caller must
// call release exactly once when the slot was granted.
func (s *Server) acquire(ctx context.Context) (time.Duration, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	started := time.Now()
	select {
	case s.slots <- struct{}{}:
		return 0, nil
	default:
	}

	s.mu.Lock()
	if s.queued >= s.policy.MaxQueue {
		s.mu.Unlock()
		return 0, errOverloaded
	}
	s.queued++
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		s.queued--
		s.mu.Unlock()
	}()

	timer := time.NewTimer(s.policy.QueueTimeout)
	defer timer.Stop()
	select {
	case s.slots <- struct{}{}:
		return time.Since(started), nil
	case <-timer.C:
		return time.Since(started), errQueueTimeout
	case <-ctx.Done():
		return time.Since(started), ctx.Err()
	}
}

// release frees a generation slot reserved by acquire.
func (s *Server) release() { <-s.slots }

// writeAdmissionError maps a rejected admission to a deterministic HTTP
// response. A client that has already gone away has nothing to send to, so
// that case writes no response and just returns.
func (s *Server) writeAdmissionError(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, errOverloaded):
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusServiceUnavailable,
			fmt.Sprintf("server at capacity: %d generations in flight, %d queued",
				s.Inflight(), s.Queued()))
	case errors.Is(err, errQueueTimeout):
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusServiceUnavailable,
			fmt.Sprintf("queue wait exceeded %s while %d generations were in flight",
				s.policy.QueueTimeout, s.Inflight()))
	case errors.Is(err, context.DeadlineExceeded):
		writeError(w, http.StatusGatewayTimeout, "request exceeded the generation timeout")
	}
}

// recordBackend remembers the backend that actually served the last
// request, so /healthz can report it without building a backend itself.
func (s *Server) recordBackend(name string) {
	if name == "" {
		return
	}
	s.mu.Lock()
	s.lastBackend = name
	s.mu.Unlock()
}

func (s *Server) backendName() string {
	s.mu.Lock()
	last := s.lastBackend
	s.mu.Unlock()
	if last != "" {
		return last
	}
	return s.model.InferenceBackendName()
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":         "ok",
		"parameters":     s.model.ParameterCount(),
		"backend":        s.backendName(),
		"max_concurrent": s.policy.MaxConcurrent,
		"max_queue":      s.policy.MaxQueue,
		"inflight":       s.Inflight(),
		"queued":         s.Queued(),
	})
}

// generateRequest is the /v1/generate schema.
type generateRequest struct {
	Prompt            string   `json:"prompt"`
	MaxTokens         int      `json:"max_tokens,omitempty"`
	Temperature       float64  `json:"temperature,omitempty"`
	TopK              int      `json:"top_k,omitempty"`
	TopP              float64  `json:"top_p,omitempty"`
	Seed              int64    `json:"seed,omitempty"`
	Stop              []string `json:"stop,omitempty"`
	RepetitionPenalty float64  `json:"repetition_penalty,omitempty"`
}

// chatRequest is the /v1/chat schema.
type chatRequest struct {
	generateRequest
	Messages []lm.ChatMessage `json:"messages"`
	Stream   bool             `json:"stream,omitempty"`
}

func clampOutput(n, cap int) int {
	if n <= 0 || n > cap {
		return cap
	}
	return n
}

func (s *Server) decode(w http.ResponseWriter, r *http.Request, v any) error {
	if r.Body == nil {
		return errors.New("empty request body")
	}
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, s.policy.MaxBodyBytes))
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	return nil
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), s.policy.RequestTimeout)
	defer cancel()
	r = r.WithContext(ctx)

	var req generateRequest
	if err := s.decode(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if strings.TrimSpace(req.Prompt) == "" {
		writeError(w, http.StatusBadRequest, "prompt is required")
		return
	}
	promptTokens := len(s.model.Tokenizer.Encode(req.Prompt))
	if promptTokens > s.policy.MaxPromptTokens {
		writeError(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("prompt has %d tokens; limit is %d", promptTokens, s.policy.MaxPromptTokens))
		return
	}
	opts := lm.GenerateOptions{
		MaxTokens:         clampOutput(req.MaxTokens, s.policy.MaxOutputTokens),
		Temperature:       req.Temperature,
		TopK:              req.TopK,
		TopP:              req.TopP,
		Seed:              req.Seed,
		RepetitionPenalty: req.RepetitionPenalty,
		Stop:              req.Stop,
	}
	queueTime, err := s.acquire(ctx)
	if err != nil {
		s.writeAdmissionError(w, err)
		return
	}
	defer s.release()

	started := time.Now()
	out, stats, err := s.model.GenerateWithContextStats(ctx, req.Prompt, opts, nil)
	generation := time.Since(started)
	s.recordBackend(stats.Backend)
	if err != nil {
		if ctx.Err() != nil {
			writeError(w, http.StatusGatewayTimeout, "request exceeded the generation timeout")
			return
		}
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	usage := Usage{GenerateStats: stats, QueueMs: queueTime.Milliseconds(), GenerationMs: generation.Milliseconds()}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-M3tal-Backend", usage.Backend)
	w.Header().Set("X-M3tal-Queue-Ms", fmt.Sprintf("%d", usage.QueueMs))
	w.Header().Set("X-M3tal-Generation-Ms", fmt.Sprintf("%d", usage.GenerationMs))
	_ = json.NewEncoder(w).Encode(map[string]any{"text": out, "usage": usage})
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), s.policy.RequestTimeout)
	defer cancel()
	r = r.WithContext(ctx)

	var req chatRequest
	if err := s.decode(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages is required")
		return
	}
	// Generation prompts don't require a trailing assistant turn, so we
	// validate structure manually: known roles, non-empty content, and at
	// least one non-assistant turn to reply to.
	hasNonAssistant := false
	for i, m := range req.Messages {
		switch m.Role {
		case lm.RoleSystem, lm.RoleUser, lm.RoleAssistant, lm.RoleTool:
		default:
			writeError(w, http.StatusBadRequest, fmt.Sprintf("message %d: unknown role %q", i, m.Role))
			return
		}
		if strings.TrimSpace(m.Content) == "" {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("message %d: empty content", i))
			return
		}
		if m.Role != lm.RoleAssistant {
			hasNonAssistant = true
		}
	}
	if !hasNonAssistant {
		writeError(w, http.StatusBadRequest, "conversation has no user/system/tool turn to reply to")
		return
	}
	conv := lm.Conversation{Messages: req.Messages}
	promptIDs := conv.EncodePrompt(s.model.Tokenizer)
	if len(promptIDs) > s.policy.MaxPromptTokens {
		writeError(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("prompt has %d tokens; limit is %d", len(promptIDs), s.policy.MaxPromptTokens))
		return
	}
	promptText := s.model.Tokenizer.Decode(promptIDs)
	opts := lm.GenerateOptions{
		MaxTokens:         clampOutput(req.MaxTokens, s.policy.MaxOutputTokens),
		Temperature:       req.Temperature,
		TopK:              req.TopK,
		TopP:              req.TopP,
		Seed:              req.Seed,
		RepetitionPenalty: req.RepetitionPenalty,
		Stop:              append(append([]string{}, req.Stop...), lm.StopMarkers()...),
	}
	queueTime, err := s.acquire(ctx)
	if err != nil {
		s.writeAdmissionError(w, err)
		return
	}
	defer s.release()

	started := time.Now()
	if req.Stream {
		s.handleChatStream(ctx, w, promptText, opts, queueTime, started)
		return
	}
	out, stats, err := s.model.GenerateWithContextStats(ctx, promptText, opts, nil)
	generation := time.Since(started)
	s.recordBackend(stats.Backend)
	if err != nil {
		if ctx.Err() != nil {
			writeError(w, http.StatusGatewayTimeout, "request exceeded the generation timeout")
			return
		}
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	usage := Usage{GenerateStats: stats, QueueMs: queueTime.Milliseconds(), GenerationMs: generation.Milliseconds()}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-M3tal-Backend", usage.Backend)
	w.Header().Set("X-M3tal-Queue-Ms", fmt.Sprintf("%d", usage.QueueMs))
	w.Header().Set("X-M3tal-Generation-Ms", fmt.Sprintf("%d", usage.GenerationMs))
	_ = json.NewEncoder(w).Encode(map[string]any{
		"text":     out,
		"template": lm.ChatTemplateVersion,
		"usage":    usage,
	})
}

// handleChatStream serves Server-Sent Events: each "data:" line carries
// {"delta": "..."}, the last carries {"usage": {...}}, and the stream ends
// with "data: [DONE]".
func (s *Server) handleChatStream(ctx context.Context, w http.ResponseWriter, prompt string, opts lm.GenerateOptions, queueTime time.Duration, started time.Time) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	writeSSE := func(v any) bool {
		data, err := json.Marshal(v)
		if err != nil {
			return false
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			return false
		}
		flusher.Flush()
		return true
	}
	_, stats, err := s.model.GenerateWithContextStats(ctx, prompt, opts, func(delta string) error {
		if !writeSSE(map[string]string{"delta": delta}) {
			return fmt.Errorf("client disconnected")
		}
		return nil
	})
	generation := time.Since(started)
	s.recordBackend(stats.Backend)
	if err != nil {
		writeSSE(map[string]string{"error": err.Error()})
	}
	writeSSE(map[string]any{"usage": Usage{
		GenerateStats: stats,
		QueueMs:       queueTime.Milliseconds(),
		GenerationMs:  generation.Milliseconds(),
	}})
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}
