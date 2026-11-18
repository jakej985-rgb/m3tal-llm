package lm

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestDecodeCheckpointMatchesFullDecode pins the layout-aware decoder to the
// whole-document decode it replaces: same header fields, same array bytes.
func TestDecodeCheckpointMatchesFullDecode(t *testing.T) {
	model, _ := smallModel(t)
	for i, p := range model.parameters {
		for j := range p.values {
			p.values[j] += float32(i*131+j) * 1e-4
			p.first[j] = float32(j%7) * 0.01
			p.second[j] = float32(j%5) * 0.02
		}
	}
	path := filepath.Join(t.TempDir(), "ckpt.json")
	if err := model.SaveCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	decoded, err := decodeCheckpoint(data)
	if err != nil {
		t.Fatal(err)
	}
	var full checkpointHeader
	if err := json.Unmarshal(data, &full); err != nil {
		t.Fatal(err)
	}
	if string(decoded.values) != string(full.Values) {
		t.Fatal("values array bytes differ from the full decode")
	}
	if !decoded.hasOptimizer || full.Optimizer == nil {
		t.Fatalf("optimizer detection: fast=%v full=%v", decoded.hasOptimizer, full.Optimizer != nil)
	}
	if string(decoded.first) != string(full.Optimizer.First) || string(decoded.second) != string(full.Optimizer.Second) {
		t.Fatal("optimizer array bytes differ from the full decode")
	}
	if decoded.header.Step != full.Step || decoded.header.Seed != full.Seed ||
		decoded.header.FormatVersion != full.FormatVersion ||
		decoded.header.TokenizerHash != full.TokenizerHash {
		t.Fatal("header metadata differs from the full decode")
	}
	if len(decoded.header.Tokens) != len(full.Tokens) || len(decoded.header.Merges) != len(full.Merges) {
		t.Fatal("header tokenizer differs from the full decode")
	}
	if decoded.header.Config != full.Config {
		t.Fatalf("config %+v, want %+v", decoded.header.Config, full.Config)
	}
}

// TestLoadCheckpointWithKeyLookalikeTokens covers a vocabulary whose token text
// equals a key the layout-aware decoder searches for; those tokens are followed
// by a comma, not a colon, so they must not be mistaken for the keys.
func TestLoadCheckpointWithKeyLookalikeTokens(t *testing.T) {
	tok := &Tokenizer{SpecialTokens: map[string]int{}, unkTokenID: -1}
	for i := 0; i < 256; i++ {
		tok.Tokens = append(tok.Tokens, string([]byte{byte(i)}))
	}
	tok.Tokens = append(tok.Tokens, "values", "first", "second", "optimizer")
	tok.reindex()
	model, err := NewModel(Config{Context: 8, Width: 8, Layers: 1, Heads: 2}, tok, 5)
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range model.parameters {
		for j := range p.values {
			p.values[j] = float32(i*97+j) * 1e-3
			p.first[j] = float32(j%3) * 0.01
			p.second[j] = float32(j%4) * 0.02
		}
	}
	path := filepath.Join(t.TempDir(), "lookalike.json")
	if err := model.SaveCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	got, want := parameterValues(loaded), parameterValues(model)
	if len(got) != len(want) {
		t.Fatalf("parameter count %d != %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("weight %d: %v != %v", i, got[i], want[i])
		}
	}
	for i, p := range model.parameters {
		for j := range p.first {
			if loaded.parameters[i].first[j] != p.first[j] || loaded.parameters[i].second[j] != p.second[j] {
				t.Fatalf("optimizer state mismatch at param %d idx %d", i, j)
			}
		}
	}
}

func BenchmarkLoadCheckpoint6M(b *testing.B) {
	const path = "../../models/m3tal-smart-6m.json"
	if _, err := os.Stat(path); err != nil {
		b.Skip("checkpoint artifact not present")
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := LoadCheckpoint(path); err != nil {
			b.Fatal(err)
		}
	}
}
