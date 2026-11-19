package lm

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"testing"
	"time"
)

// TestConcurrentGenerate guards the lazily built inference backend: several
// callers may generate from one model at the same time (the HTTP server
// shares a single *Model across handlers). Run under -race.
func TestConcurrentGenerate(t *testing.T) {
	tok, err := TrainTokenizer([]string{"hello world hello there"}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 16, Width: 16, Layers: 2, Heads: 2}, tok, 1)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := model.GenerateWithOptions("hello", GenerateOptions{MaxTokens: 4}); err != nil {
				t.Errorf("generate: %v", err)
			}
		}()
	}
	wg.Wait()
}

// TestGenerateContextCancelStopsGeneration checks that cancellation is
// observed at a token boundary instead of running to MaxTokens.
func TestGenerateContextCancelStopsGeneration(t *testing.T) {
	tok, err := TrainTokenizer([]string{"hello world hello there"}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 16, Width: 16, Layers: 1, Heads: 2}, tok, 1)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := model.GenerateWithContext(ctx, "hello", GenerateOptions{MaxTokens: 64}, nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

// TestGenerateContextCancelMidGeneration cancels from inside the stream
// callback, so cancellation lands after generation has started: generation
// must stop at the next token boundary, return the text produced so far, and
// report context.Canceled rather than running to MaxTokens.
func TestGenerateContextCancelMidGeneration(t *testing.T) {
	tok, err := TrainTokenizer([]string{"hello world hello there"}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 16, Width: 16, Layers: 1, Heads: 2}, tok, 1)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	deltas := 0
	out, err := model.GenerateWithContext(ctx, "hello", GenerateOptions{MaxTokens: 128}, func(delta string) error {
		deltas++
		cancel()
		return nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if deltas != 1 {
		t.Fatalf("expected generation to stop after the first delta, got %d", deltas)
	}
	if out == "" {
		t.Fatal("expected the already-generated text to be returned with the cancellation")
	}
}

// TestCancelledGenerationLeavesNoGoroutines checks that repeated cancelled
// generations release their per-request state (KV cache, backend scratch)
// instead of leaking goroutines.
func TestCancelledGenerationLeavesNoGoroutines(t *testing.T) {
	tok, err := TrainTokenizer([]string{"hello world hello there"}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 16, Width: 16, Layers: 1, Heads: 2}, tok, 1)
	if err != nil {
		t.Fatal(err)
	}
	// Warm the lazily created inference backend before taking the baseline.
	if _, err := model.GenerateWithOptions("hello", GenerateOptions{MaxTokens: 2}); err != nil {
		t.Fatal(err)
	}
	baseline := runtime.NumGoroutine()
	for i := 0; i < 32; i++ {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := model.GenerateWithContext(ctx, "hello", GenerateOptions{MaxTokens: 64}, nil); !errors.Is(err, context.Canceled) {
			t.Fatalf("iteration %d: expected context.Canceled, got %v", i, err)
		}
	}
	if n := settleGoroutines(baseline, 2*time.Second); n > baseline {
		t.Fatalf("goroutines grew from %d to %d across cancelled generations", baseline, n)
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
