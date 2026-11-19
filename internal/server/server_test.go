package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"m3tal-llm/internal/lm"
)

func testServer(t *testing.T, policy Policy) (*Server, *httptest.Server) {
	t.Helper()
	tok := testTokenizer(t)
	cfg := lm.Config{Context: 32, Width: 32, Layers: 2, Heads: 4}
	model, err := lm.NewModel(cfg, tok, 1)
	if err != nil {
		t.Fatal(err)
	}
	srv := New(model, policy)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return srv, ts
}

func testTokenizer(t *testing.T) *lm.Tokenizer {
	t.Helper()
	tok, err := lm.TrainTokenizer([]string{
		"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>\n",
		"The quick brown fox jumps over the lazy dog.",
	}, 256)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func post(t *testing.T, url string, body any) (*http.Response, []byte) {
	t.Helper()
	data, _ := json.Marshal(body)
	resp, err := http.Post(url, "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := new(bytes.Buffer)
	_, _ = buf.ReadFrom(resp.Body)
	return resp, buf.Bytes()
}

func TestHealth(t *testing.T) {
	_, ts := testServer(t, Policy{})
	resp, err := http.Get(ts.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestGenerateEndpoint(t *testing.T) {
	_, ts := testServer(t, Policy{})
	resp, body := post(t, ts.URL+"/v1/generate", map[string]any{
		"prompt": "Hello", "max_tokens": 4, "seed": 1,
	})
	if resp.StatusCode != 200 {
		t.Fatalf("status %d body %s", resp.StatusCode, body)
	}
	var out struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	if out.Text == "" {
		t.Fatal("empty generation")
	}
}

func TestChatEndpoint(t *testing.T) {
	_, ts := testServer(t, Policy{})
	resp, body := post(t, ts.URL+"/v1/chat", map[string]any{
		"messages": []map[string]string{
			{"role": "user", "content": "Hello"},
		},
		"max_tokens": 4, "seed": 1,
	})
	if resp.StatusCode != 200 {
		t.Fatalf("status %d body %s", resp.StatusCode, body)
	}
	var out struct {
		Text     string `json:"text"`
		Template string `json:"template"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatal(err)
	}
	if out.Template != lm.ChatTemplateVersion {
		t.Fatalf("template %q", out.Template)
	}
	if strings.Contains(out.Text, "<|im_end|>") {
		t.Fatalf("stop marker leaked: %q", out.Text)
	}
}

func TestChatValidationErrors(t *testing.T) {
	_, ts := testServer(t, Policy{})
	cases := []map[string]any{
		{"messages": []any{}}, // empty
		{"messages": []map[string]string{{"role": "user", "content": ""}}},        // invalid conv
		{"messages": []map[string]string{{"role": "assistant", "content": "hi"}}}, // no non-assistant
	}
	for i, body := range cases {
		resp, respBody := post(t, ts.URL+"/v1/chat", body)
		if resp.StatusCode != 400 {
			t.Fatalf("case %d: status %d body %s", i, resp.StatusCode, respBody)
		}
	}
}

func TestPromptTokenLimit(t *testing.T) {
	_, ts := testServer(t, Policy{MaxPromptTokens: 8})
	longPrompt := strings.Repeat("word ", 100)
	resp, body := post(t, ts.URL+"/v1/generate", map[string]any{
		"prompt": longPrompt, "max_tokens": 2,
	})
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("status %d body %s", resp.StatusCode, body)
	}
}

func TestOutputTokenClamp(t *testing.T) {
	srv, ts := testServer(t, Policy{MaxOutputTokens: 3})
	if srv.Policy().MaxOutputTokens != 3 {
		t.Fatal("policy not applied")
	}
	resp, _ := post(t, ts.URL+"/v1/generate", map[string]any{
		"prompt": "Hi", "max_tokens": 100,
	})
	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestMalformedJSON(t *testing.T) {
	_, ts := testServer(t, Policy{})
	resp, err := http.Post(ts.URL+"/v1/generate", "application/json",
		strings.NewReader("{not json"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 400 {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestOversizedBody(t *testing.T) {
	_, ts := testServer(t, Policy{MaxBodyBytes: 128})
	big := map[string]any{"prompt": strings.Repeat("x", 1000)}
	resp, _ := post(t, ts.URL+"/v1/generate", big)
	if resp.StatusCode != 400 {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestChatStreamSSE(t *testing.T) {
	_, ts := testServer(t, Policy{})
	data, _ := json.Marshal(map[string]any{
		"messages":   []map[string]string{{"role": "user", "content": "Hi"}},
		"max_tokens": 3, "stream": true, "seed": 2,
	})
	resp, err := http.Post(ts.URL+"/v1/chat", "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type %q", ct)
	}
	body := new(bytes.Buffer)
	_, _ = body.ReadFrom(resp.Body)
	text := body.String()
	if !strings.Contains(text, "data: ") || !strings.Contains(text, "[DONE]") {
		t.Fatalf("bad SSE stream: %q", text)
	}
}

// TestChatStreamDisconnectStopsGeneration proves an SSE client disconnect
// reaches model generation: after the client cancels and closes the stream,
// the handler must return instead of generating the full output cap. The cap
// is far larger than can be generated in the wait window, so the handler
// returning is evidence of cancellation, not of an early finish.
func TestChatStreamDisconnectStopsGeneration(t *testing.T) {
	tok := testTokenizer(t)
	model, err := lm.NewModel(lm.Config{Context: 32, Width: 32, Layers: 2, Heads: 4}, tok, 1)
	if err != nil {
		t.Fatal(err)
	}
	srv := New(model, Policy{MaxOutputTokens: 1_000_000, RequestTimeout: 10 * time.Second})

	handled := make(chan struct{})
	var once sync.Once
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		srv.Handler().ServeHTTP(w, r)
		once.Do(func() { close(handled) })
	}))
	defer ts.Close()

	baseline := runtime.NumGoroutine()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	data, _ := json.Marshal(map[string]any{
		"messages":   []map[string]string{{"role": "user", "content": "Hi"}},
		"max_tokens": 1_000_000,
		"stream":     true,
		"seed":       3,
	})
	req, err := http.NewRequestWithContext(ctx, "POST", ts.URL+"/v1/chat", bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}

	// Read up to the first delta so the handler is provably mid-generation.
	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			t.Fatalf("read SSE before first delta: %v", err)
		}
		if strings.Contains(line, "\"delta\"") {
			break
		}
	}
	cancel()
	_ = resp.Body.Close()

	select {
	case <-handled:
	case <-time.After(5 * time.Second):
		t.Fatal("generation kept running after the client disconnected")
	}
	if n := settleGoroutines(baseline+2, 5*time.Second); n > baseline+2 {
		t.Fatalf("goroutines still running after disconnect: %d (baseline %d)", n, baseline)
	}
}

func settleGoroutines(target int, timeout time.Duration) int {
	deadline := time.Now().Add(timeout)
	n := runtime.NumGoroutine()
	for n > target && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
		n = runtime.NumGoroutine()
	}
	return n
}

// TestRequestTimeoutEnforced verifies Policy.RequestTimeout actually stops
// generation: an already-expired deadline must fail fast, not generate.
func TestRequestTimeoutEnforced(t *testing.T) {
	_, ts := testServer(t, Policy{RequestTimeout: time.Nanosecond, MaxOutputTokens: 64})
	time.Sleep(2 * time.Millisecond)
	resp, body := post(t, ts.URL+"/v1/generate", map[string]any{
		"prompt": "Hello", "max_tokens": 64,
	})
	if resp.StatusCode != http.StatusGatewayTimeout {
		t.Fatalf("status %d body %s", resp.StatusCode, body)
	}
}

func TestDefaultPolicyApplied(t *testing.T) {
	srv, _ := testServer(t, Policy{})
	p := srv.Policy()
	if p.MaxPromptTokens != DefaultPolicy.MaxPromptTokens ||
		p.MaxOutputTokens != DefaultPolicy.MaxOutputTokens ||
		p.MaxBodyBytes != DefaultPolicy.MaxBodyBytes ||
		p.RequestTimeout != DefaultPolicy.RequestTimeout {
		t.Fatalf("defaults not applied: %+v", p)
	}
}
