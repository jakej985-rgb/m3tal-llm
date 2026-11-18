package lm

import (
	"fmt"
	"math"
)

// This file holds the integrity checks applied before a checkpoint is trusted:
// model shape, tokenizer state, and the finiteness of weights and optimizer
// moments. They run on load (before anything is computed with the state) and
// on save (so a broken model is never written in the first place).

// checkpointFormatVersion is the newest JSON checkpoint container version this
// build writes and reads.
const checkpointFormatVersion = 2

// Validate reports whether the config describes a supported model shape.
// Layers and Heads of 0 mean "use the default" (older checkpoints omitted
// them); negative values are rejected rather than silently defaulted.
func (c Config) Validate() error {
	if c.Context < 2 || c.Context > 2048 {
		return fmt.Errorf("config: context must be between 2 and 2048, got %d", c.Context)
	}
	if c.Width < 2 || c.Width > 2048 {
		return fmt.Errorf("config: width must be between 2 and 2048, got %d", c.Width)
	}
	if c.Layers < 0 || c.Layers > 16 {
		return fmt.Errorf("config: layers must be between 1 and 16, got %d", c.Layers)
	}
	if c.Heads < 0 || c.Heads > 32 {
		return fmt.Errorf("config: heads must be between 1 and 32, got %d", c.Heads)
	}
	normalized := c
	normalized.Normalize()
	if normalized.Width%normalized.Heads != 0 {
		return fmt.Errorf("config: width %d must be divisible by heads %d", normalized.Width, normalized.Heads)
	}
	return nil
}

// Validate reports whether the tokenizer tables are self-consistent: a
// vocabulary within the supported range, no empty or duplicate token strings,
// merges that reference existing tokens, and special tokens pointing at
// existing ids. A checkpoint that fails here would mis-tokenize silently, so
// loading rejects it.
func (t *Tokenizer) Validate() error {
	if t == nil {
		return fmt.Errorf("tokenizer: nil")
	}
	if n := len(t.Tokens); n < MinVocabSize || n > MaxVocabSize {
		return fmt.Errorf("tokenizer: vocabulary has %d tokens, want %d..%d", n, MinVocabSize, MaxVocabSize)
	}
	seen := make(map[string]int, len(t.Tokens))
	for i, token := range t.Tokens {
		if token == "" {
			return fmt.Errorf("tokenizer: token %d is the empty string", i)
		}
		if prev, dup := seen[token]; dup {
			return fmt.Errorf("tokenizer: token %d duplicates token %d (%q)", i, prev, token)
		}
		seen[token] = i
	}
	for i, merge := range t.Merges {
		if _, ok := seen[merge.Left]; !ok {
			return fmt.Errorf("tokenizer: merge %d left operand %q is not a token", i, merge.Left)
		}
		if _, ok := seen[merge.Right]; !ok {
			return fmt.Errorf("tokenizer: merge %d right operand %q is not a token", i, merge.Right)
		}
		if _, ok := seen[merge.Token]; !ok {
			return fmt.Errorf("tokenizer: merge %d result %q is not a token", i, merge.Token)
		}
	}
	byID := make(map[int]string, len(t.SpecialTokens))
	for name, id := range t.SpecialTokens {
		if name == "" {
			return fmt.Errorf("tokenizer: special token with empty name")
		}
		if id < 0 || id >= len(t.Tokens) {
			return fmt.Errorf("tokenizer: special token %q has id %d outside 0..%d", name, id, len(t.Tokens)-1)
		}
		if other, dup := byID[id]; dup && other != name {
			return fmt.Errorf("tokenizer: special tokens %q and %q share id %d", other, name, id)
		}
		byID[id] = name
		if t.Tokens[id] != name {
			return fmt.Errorf("tokenizer: special token %q points at token %q", name, t.Tokens[id])
		}
	}
	return nil
}

// validateModelState rejects state that cannot be trained or evaluated: a
// negative step, non-finite weights or Adam moments, optimizer arrays whose
// lengths disagree with the parameter, and negative second moments (which are
// squared gradients and cannot be negative).
//
// It scans every parameter, so it runs once per load or save rather than per
// step.
func (m *Model) validateModelState() error {
	if m == nil {
		return fmt.Errorf("model is nil")
	}
	if m.step < 0 {
		return fmt.Errorf("step %d is negative", m.step)
	}
	for _, s := range m.parameterSpecs() {
		if err := requireFinite(s.Name, s.P.values); err != nil {
			return err
		}
		if len(s.P.first) != len(s.P.values) || len(s.P.second) != len(s.P.values) {
			return fmt.Errorf("%s: optimizer state has %d first and %d second moments for %d values",
				s.Name, len(s.P.first), len(s.P.second), len(s.P.values))
		}
		if err := requireFinite("optimizer.first."+s.Name, s.P.first); err != nil {
			return err
		}
		if err := requireFinite("optimizer.second."+s.Name, s.P.second); err != nil {
			return err
		}
		for i, v := range s.P.second {
			if v < 0 {
				return fmt.Errorf("optimizer.second.%s[%d] is %v; second moments are squared gradients and cannot be negative", s.Name, i, v)
			}
		}
	}
	return nil
}

// requireFinite rejects NaN and infinities in a saved tensor.
func requireFinite(name string, values []float32) error {
	for i, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("%s[%d] is %v; weights and optimizer moments must be finite", name, i, v)
		}
	}
	return nil
}
