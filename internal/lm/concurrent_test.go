package lm

import (
	"context"
	"errors"
	"sync"
	"testing"
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
