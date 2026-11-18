package lm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// largeModel returns a model above safetensorsCheckpointMinParams but small
// enough to build quickly.
func largeModel(t *testing.T) *Model {
	t.Helper()
	tok, err := TrainTokenizer([]string{"alpha beta gamma delta epsilon zeta eta theta. "}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 16, Width: 128, Layers: 6, Heads: 8}, tok, 3)
	if err != nil {
		t.Fatal(err)
	}
	if model.ParameterCount() < safetensorsCheckpointMinParams {
		t.Fatalf("test model has %d parameters, needs at least %d", model.ParameterCount(), safetensorsCheckpointMinParams)
	}
	return model
}

func perturbModel(model *Model) {
	for i, p := range model.parameters {
		for j := range p.values {
			p.values[j] = float32(i*71+j) * 1e-4
			p.first[j] = float32(j%9) * 0.01
			p.second[j] = float32(j%5) * 0.02
		}
	}
	model.step = 4321
	model.seed = 77
}

func assertSameModel(t *testing.T, got, want *Model) {
	t.Helper()
	gotValues, wantValues := parameterValues(got), parameterValues(want)
	if len(gotValues) != len(wantValues) {
		t.Fatalf("parameter count %d != %d", len(gotValues), len(wantValues))
	}
	for i := range wantValues {
		if gotValues[i] != wantValues[i] {
			t.Fatalf("weight %d: %v != %v", i, gotValues[i], wantValues[i])
		}
	}
	for i, p := range want.parameters {
		for j := range p.first {
			if got.parameters[i].first[j] != p.first[j] || got.parameters[i].second[j] != p.second[j] {
				t.Fatalf("optimizer state mismatch at param %d idx %d", i, j)
			}
		}
	}
	if got.step != want.step || got.seed != want.seed {
		t.Fatalf("state: step=%d seed=%d, want step=%d seed=%d", got.step, got.seed, want.step, want.seed)
	}
	if len(got.Tokenizer.Tokens) != len(want.Tokenizer.Tokens) {
		t.Fatalf("tokenizer size %d != %d", len(got.Tokenizer.Tokens), len(want.Tokenizer.Tokens))
	}
	for i := range want.Tokenizer.Tokens {
		if got.Tokenizer.Tokens[i] != want.Tokenizer.Tokens[i] {
			t.Fatalf("token %d: %q != %q", i, got.Tokenizer.Tokens[i], want.Tokenizer.Tokens[i])
		}
	}
	if len(got.Tokenizer.Merges) != len(want.Tokenizer.Merges) {
		t.Fatalf("merge count %d != %d", len(got.Tokenizer.Merges), len(want.Tokenizer.Merges))
	}
}

// TestSaveCheckpointUsesSafetensorsForLargeModels checks the container choice,
// the round trip through it, and that the path extension does not matter
// because LoadCheckpoint detects the container from its bytes.
func TestSaveCheckpointUsesSafetensorsForLargeModels(t *testing.T) {
	model := largeModel(t)
	perturbModel(model)
	dir := t.TempDir()

	big := filepath.Join(dir, "large.json") // .json path, safetensors content
	if err := model.SaveCheckpoint(big); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(big)
	if err != nil {
		t.Fatal(err)
	}
	if !isSafetensorsContainer(data) {
		t.Fatal("large checkpoint was not written in safetensors format")
	}
	if strings.HasPrefix(strings.TrimSpace(string(data[:16])), "{") {
		t.Fatal("large checkpoint still looks like JSON")
	}
	loaded, err := LoadCheckpoint(big)
	if err != nil {
		t.Fatal(err)
	}
	assertSameModel(t, loaded, model)

	// A small model still writes inspectable JSON.
	small, _ := smallModel(t)
	smallPath := filepath.Join(dir, "small.json")
	if err := small.SaveCheckpoint(smallPath); err != nil {
		t.Fatal(err)
	}
	smallData, err := os.ReadFile(smallPath)
	if err != nil {
		t.Fatal(err)
	}
	if isSafetensorsContainer(smallData) {
		t.Fatal("small checkpoint should stay JSON")
	}
	if !strings.HasPrefix(string(smallData), "{") {
		t.Fatal("small checkpoint is not JSON")
	}
	if _, err := LoadCheckpoint(smallPath); err != nil {
		t.Fatal(err)
	}
}

// TestSafetensorsCheckpointMatchesJSON checks that both containers of the same
// model load to identical state.
func TestSafetensorsCheckpointMatchesJSON(t *testing.T) {
	model := largeModel(t)
	perturbModel(model)
	dir := t.TempDir()

	stPath := filepath.Join(dir, "model.safetensors")
	if err := model.SaveCheckpoint(stPath); err != nil {
		t.Fatal(err)
	}
	jsonPath := filepath.Join(dir, "model.json")
	if err := model.saveJSONCheckpoint(jsonPath); err != nil {
		t.Fatal(err)
	}
	fromSafetensors, err := LoadCheckpoint(stPath)
	if err != nil {
		t.Fatal(err)
	}
	fromJSON, err := LoadCheckpoint(jsonPath)
	if err != nil {
		t.Fatal(err)
	}
	assertSameModel(t, fromSafetensors, fromJSON)
}

func TestSafetensorsCheckpointRejectsCorruption(t *testing.T) {
	model := largeModel(t)
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := model.SaveCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	truncated := filepath.Join(t.TempDir(), "truncated.safetensors")
	if err := os.WriteFile(truncated, data[:len(data)/2], 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(truncated); err == nil {
		t.Fatal("expected error for truncated safetensors checkpoint")
	}

	weightsOnly := filepath.Join(t.TempDir(), "weights.safetensors")
	if err := model.SaveSafetensors(weightsOnly); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(weightsOnly); err == nil {
		t.Fatal("expected error for a weights-only safetensors file")
	}
}
