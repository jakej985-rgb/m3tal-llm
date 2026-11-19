package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

// holdSlot occupies a generation slot directly, so admission behaviour can
// be tested deterministically without depending on how long a request takes.
func holdSlot(t *testing.T, srv *Server) func() {
	t.Helper()
	if _, err := srv.acquire(context.Background()); err != nil {
		t.Fatalf("acquire: %v", err)
	}
	var once sync.Once
	return func() { once.Do(srv.release) }
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(2 * time.Millisecond)
	}
}

type usagePayload struct {
	SelectedBackend string `json:"selected_backend"`
	Backend         string `json:"backend"`
	PromptTokens    int    `json:"prompt_tokens"`
	OutputTokens    int    `json:"output_tokens"`
	FallbackCount   int64  `json:"fallback_count"`
	TransferCount   int64  `json:"transfer_count"`
	TransferBytes   int64  `json:"transfer_bytes"`
	QueueMs         int64  `json:"queue_ms"`
	GenerationMs    int64  `json:"generation_ms"`
}

func TestOverloadReturns503(t *testing.T) {
	srv, ts := testServer(t, Policy{MaxConcurrent: 1, MaxQueue: -1, MaxOutputTokens: 8})
	release := holdSlot(t, srv)
	defer release()

	resp, body := post(t, ts.URL+"/v1/generate", map[string]any{"prompt": "Hello", "max_tokens": 2})
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status %d body %s, want 503", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "at capacity") {
		t.Fatalf("body %s does not explain the overload", body)
	}
	if got := resp.Header.Get("Retry-After"); got != "1" {
		t.Fatalf("Retry-After %q, want \"1\"", got)
	}
	if srv.Queued() != 0 {
		t.Fatalf("rejected request stayed queued: %d", srv.Queued())
	}
}

func TestQueueTimeoutReturns503(t *testing.T) {
	srv, ts := testServer(t, Policy{
		MaxConcurrent:   1,
		MaxQueue:        2,
		QueueTimeout:    30 * time.Millisecond,
		MaxOutputTokens: 8,
	})
	release := holdSlot(t, srv)
	defer release()

	started := time.Now()
	resp, body := post(t, ts.URL+"/v1/generate", map[string]any{"prompt": "Hello", "max_tokens": 2})
	elapsed := time.Since(started)
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status %d body %s, want 503", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "queue wait exceeded") {
		t.Fatalf("body %s does not explain the queue timeout", body)
	}
	if elapsed < 20*time.Millisecond {
		t.Fatalf("queue timeout returned after %v; it did not wait", elapsed)
	}
	if srv.Queued() != 0 {
		t.Fatalf("timed-out request stayed queued: %d", srv.Queued())
	}
}

func TestCanceledQueuedRequestFreesQueue(t *testing.T) {
	srv, ts := testServer(t, Policy{
		MaxConcurrent:   1,
		MaxQueue:        2,
		QueueTimeout:    5 * time.Second,
		MaxOutputTokens: 8,
	})
	release := holdSlot(t, srv)
	defer release()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		data, _ := json.Marshal(map[string]any{"prompt": "Hello", "max_tokens": 2})
		req, err := http.NewRequestWithContext(ctx, "POST", ts.URL+"/v1/generate", bytes.NewReader(data))
		if err != nil {
			done <- err
			return
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := http.DefaultClient.Do(req)
		if resp != nil {
			resp.Body.Close()
		}
		done <- err
	}()

	waitFor(t, "request to queue", func() bool { return srv.Queued() == 1 })
	cancel()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("canceled queued request returned no error")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("canceled queued request did not return")
	}
	waitFor(t, "queue to drain", func() bool { return srv.Queued() == 0 })

	release()
	resp, body := post(t, ts.URL+"/v1/generate", map[string]any{"prompt": "Hello", "max_tokens": 2})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("request after cancellation: status %d body %s", resp.StatusCode, body)
	}
	if srv.Inflight() != 0 {
		t.Fatalf("slot leaked: %d in flight", srv.Inflight())
	}
}

func TestParallelRequestsAllAdmittedWithinLimits(t *testing.T) {
	srv, ts := testServer(t, Policy{
		MaxConcurrent:   2,
		MaxQueue:        8,
		QueueTimeout:    5 * time.Second,
		MaxOutputTokens: 4,
	})

	const requests = 8
	var wg sync.WaitGroup
	statuses := make(chan int, requests)
	for i := 0; i < requests; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			data, _ := json.Marshal(map[string]any{
				"prompt": "Hello", "max_tokens": 4, "seed": 1,
			})
			resp, err := http.Post(ts.URL+"/v1/generate", "application/json", bytes.NewReader(data))
			if err != nil {
				statuses <- -1
				return
			}
			defer resp.Body.Close()
			statuses <- resp.StatusCode
		}()
	}
	wg.Wait()
	close(statuses)

	for status := range statuses {
		if status != http.StatusOK {
			t.Fatalf("parallel request status %d, want 200", status)
		}
	}
	if srv.Inflight() != 0 || srv.Queued() != 0 {
		t.Fatalf("slots leaked: inflight=%d queued=%d", srv.Inflight(), srv.Queued())
	}
}

func TestGenerateUsageTelemetry(t *testing.T) {
	_, ts := testServer(t, Policy{MaxOutputTokens: 8})
	resp, body := post(t, ts.URL+"/v1/generate", map[string]any{
		"prompt": "Hello", "max_tokens": 4, "seed": 1,
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d body %s", resp.StatusCode, body)
	}
	var out struct {
		Text  string       `json:"text"`
		Usage usagePayload `json:"usage"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("decode usage: %v (%s)", err, body)
	}
	u := out.Usage
	if u.Backend != "cpu" || u.SelectedBackend != "cpu" {
		t.Fatalf("backend %q selected %q, want cpu/cpu", u.Backend, u.SelectedBackend)
	}
	if u.PromptTokens <= 0 {
		t.Fatalf("prompt tokens %d, want > 0", u.PromptTokens)
	}
	if u.OutputTokens <= 0 || u.OutputTokens > 4 {
		t.Fatalf("output tokens %d, want 1..4", u.OutputTokens)
	}
	if u.QueueMs < 0 || u.GenerationMs < 0 {
		t.Fatalf("negative timing: queue %d generation %d", u.QueueMs, u.GenerationMs)
	}
	if u.FallbackCount != 0 {
		t.Fatalf("fallback count %d, want 0 for a short prompt", u.FallbackCount)
	}
	if u.TransferCount != 0 || u.TransferBytes != 0 {
		t.Fatalf("CPU backend reported transfers %d/%d", u.TransferCount, u.TransferBytes)
	}
	if got := resp.Header.Get("X-M3tal-Backend"); got != "cpu" {
		t.Fatalf("X-M3tal-Backend %q, want cpu", got)
	}
	if resp.Header.Get("X-M3tal-Queue-Ms") == "" || resp.Header.Get("X-M3tal-Generation-Ms") == "" {
		t.Fatalf("missing timing headers: %v", resp.Header)
	}
}

func TestStreamEmitsUsage(t *testing.T) {
	_, ts := testServer(t, Policy{MaxOutputTokens: 8})
	data, _ := json.Marshal(map[string]any{
		"messages":   []map[string]string{{"role": "user", "content": "Hi"}},
		"max_tokens": 4,
		"stream":     true,
		"seed":       3,
	})
	resp, err := http.Post(ts.URL+"/v1/chat", "application/json", bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var usage usagePayload
	sawUsage, sawDone := false, false
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			sawDone = true
			break
		}
		var event struct {
			Usage *usagePayload `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &event); err == nil && event.Usage != nil {
			usage = *event.Usage
			sawUsage = true
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	if !sawUsage {
		t.Fatal("stream ended without a usage event")
	}
	if usage.Backend != "cpu" {
		t.Fatalf("usage backend %q, want cpu", usage.Backend)
	}
	if usage.PromptTokens <= 0 || usage.OutputTokens <= 0 {
		t.Fatalf("usage tokens prompt=%d output=%d", usage.PromptTokens, usage.OutputTokens)
	}
	if !sawDone {
		t.Fatal("stream ended without [DONE]")
	}
}

func TestHealthReportsScheduling(t *testing.T) {
	srv, ts := testServer(t, Policy{MaxConcurrent: 3, MaxQueue: 5})
	get := func() map[string]any {
		resp, err := http.Get(ts.URL + "/healthz")
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatal(err)
		}
		return out
	}

	health := get()
	if health["backend"] != "unloaded" {
		t.Fatalf("backend before generation = %v, want unloaded", health["backend"])
	}
	if health["max_concurrent"] != float64(3) || health["max_queue"] != float64(5) {
		t.Fatalf("scheduling limits not reported: %v", health)
	}
	if _, err := srv.model.Generate("hi", 1, 0, 1); err != nil {
		t.Fatal(err)
	}
	if got := get()["backend"]; got != "cpu" {
		t.Fatalf("backend after generation = %v, want cpu", got)
	}
}

// TestShutdownReleasesSlots simulates a server shutdown: closing the client
// connections cancels the request contexts, so both the in-flight streaming
// request and the queued request must return and release their slots.
func TestShutdownReleasesSlots(t *testing.T) {
	srv, ts := testServer(t, Policy{
		MaxConcurrent:   1,
		MaxQueue:        4,
		QueueTimeout:    5 * time.Second,
		MaxOutputTokens: 1_000_000,
		RequestTimeout:  30 * time.Second,
	})

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
	waitFor(t, "streaming request to hold the slot", func() bool { return srv.Inflight() == 1 })

	queued := make(chan error, 1)
	go func() {
		body, _ := json.Marshal(map[string]any{"prompt": "Hello", "max_tokens": 2})
		r, err := http.Post(ts.URL+"/v1/generate", "application/json", bytes.NewReader(body))
		if r != nil {
			r.Body.Close()
		}
		queued <- err
	}()
	waitFor(t, "request to queue", func() bool { return srv.Queued() == 1 })

	ts.CloseClientConnections()
	_ = resp.Body.Close()
	cancel()

	select {
	case <-queued:
	case <-time.After(5 * time.Second):
		t.Fatal("queued request did not return after shutdown")
	}
	waitFor(t, "slots to be released", func() bool {
		return srv.Inflight() == 0 && srv.Queued() == 0
	})
}
