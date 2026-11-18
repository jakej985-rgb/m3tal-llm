package lm

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// exportLegacyValues renders a current model in a legacy container's layout, so
// an import can be round-tripped. The legacy feed-forward matrices are the
// current gate (input projection) and down (output projection).
func exportLegacyValues(m *Model, layout LegacyLayout) []float32 {
	cfg := m.Config
	cfg.Normalize()
	w, context := cfg.Width, cfg.Context
	out := append([]float32{}, m.tokenEmb.values...)
	if layout.PositionalEmbeddings {
		for i := 0; i < context*w; i++ {
			out = append(out, float32(i%11)*0.01)
		}
	}
	for _, b := range m.blocks {
		if layout.NormGammas {
			out = append(out, b.normAttn.values...)
			out = append(out, b.normFF.values...)
		}
		for _, l := range []linear{b.query, b.key, b.value, b.attnOut} {
			out = append(out, l.weights.values...)
			out = append(out, l.bias.values...)
		}
		out = append(out, b.ffGate.weights.values...)
		out = append(out, b.ffGate.bias.values...)
		out = append(out, b.ffDown.weights.values...)
		out = append(out, b.ffDown.bias.values...)
	}
	out = append(out, m.output.weights.values...)
	out = append(out, m.output.bias.values...)
	return out
}

func equalF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestDetectLegacyLayout(t *testing.T) {
	m, _ := smallModel(t)
	cfg, vocab := m.Config, len(m.Tokenizer.Tokens)

	for _, layout := range legacyLayouts() {
		count := legacyParameterCount(cfg, vocab, layout)
		got, ok := DetectLegacyLayout(cfg, vocab, count)
		if !ok || got.Name != layout.Name {
			t.Fatalf("%s (%d values) not detected, got %+v ok=%v", layout.Name, count, got, ok)
		}
	}
	if _, ok := DetectLegacyLayout(cfg, vocab, m.ParameterCount()); ok {
		t.Fatal("the current layout was reported as legacy")
	}
	if _, ok := DetectLegacyLayout(cfg, vocab, 12345); ok {
		t.Fatal("an unknown parameter count was reported as legacy")
	}
}

func TestImportLegacyMapsTensors(t *testing.T) {
	for _, layout := range legacyLayouts() {
		t.Run(layout.Name, func(t *testing.T) {
			src, _ := smallModel(t)
			perturbForIntegrity(src)
			values := exportLegacyValues(src, layout)

			dst, _ := smallModel(t)
			if err := dst.ImportLegacy(layout, values); err != nil {
				t.Fatal(err)
			}

			if !equalF32(dst.tokenEmb.values, src.tokenEmb.values) {
				t.Fatal("token embedding not mapped")
			}
			if !equalF32(dst.output.weights.values, src.output.weights.values) ||
				!equalF32(dst.output.bias.values, src.output.bias.values) {
				t.Fatal("output projection not mapped")
			}
			for i := range src.blocks {
				sb, db := src.blocks[i], dst.blocks[i]
				for _, pair := range []struct {
					name string
					want []float32
					got  []float32
				}{
					{"query.weights", sb.query.weights.values, db.query.weights.values},
					{"query.bias", sb.query.bias.values, db.query.bias.values},
					{"key.weights", sb.key.weights.values, db.key.weights.values},
					{"value.weights", sb.value.weights.values, db.value.weights.values},
					{"attnOut.weights", sb.attnOut.weights.values, db.attnOut.weights.values},
					{"gate.weights", sb.ffGate.weights.values, db.ffGate.weights.values},
					{"gate.bias", sb.ffGate.bias.values, db.ffGate.bias.values},
					{"down.weights", sb.ffDown.weights.values, db.ffDown.weights.values},
					{"down.bias", sb.ffDown.bias.values, db.ffDown.bias.values},
				} {
					if !equalF32(pair.got, pair.want) {
						t.Fatalf("block %d %s not mapped", i, pair.name)
					}
				}
				if layout.NormGammas {
					if !equalF32(db.normAttn.values, sb.normAttn.values) || !equalF32(db.normFF.values, sb.normFF.values) {
						t.Fatalf("block %d norm gammas not mapped", i)
					}
				} else {
					for _, v := range append(append([]float32{}, db.normAttn.values...), db.normFF.values...) {
						if v != 1 {
							t.Fatalf("block %d gamma initialized to %v, want identity 1", i, v)
						}
					}
				}
				// SwiGLU's up branch is pinned to 1 so the block computes
				// down(silu(gate(x))).
				for _, v := range db.ffUp.weights.values {
					if v != 0 {
						t.Fatalf("block %d up weights not zeroed: %v", i, v)
					}
				}
				for _, v := range db.ffUp.bias.values {
					if v != 1 {
						t.Fatalf("block %d up bias not 1: %v", i, v)
					}
				}
			}
			if dst.step != 0 || dst.hasOptimizerMoments() {
				t.Fatalf("import should reset step/optimizer: step=%d", dst.step)
			}
		})
	}
}

func TestImportLegacyRejectsWrongLength(t *testing.T) {
	m, _ := smallModel(t)
	values := exportLegacyValues(m, legacyLayouts()[0])
	if err := m.ImportLegacy(legacyLayouts()[0], values[:len(values)-1]); err == nil {
		t.Fatal("expected a length mismatch error")
	}
	if err := m.ImportLegacy(legacyLayouts()[1], values); err == nil {
		t.Fatal("expected a layout mismatch error")
	}
}

// writeLegacyCheckpoint renders a model as a pre-current-era JSON container.
func writeLegacyCheckpoint(t *testing.T, path string, m *Model, layout LegacyLayout) []float32 {
	t.Helper()
	values := exportLegacyValues(m, layout)
	valuesJSON, err := MarshalLegacyValuesJSON(values)
	if err != nil {
		t.Fatal(err)
	}
	tokens := make([][]byte, len(m.Tokenizer.Tokens))
	for i, tok := range m.Tokenizer.Tokens {
		tokens[i] = []byte(tok)
	}
	merges := make([]checkpointMerge, len(m.Tokenizer.Merges))
	for i, merge := range m.Tokenizer.Merges {
		merges[i] = checkpointMerge{Left: []byte(merge.Left), Right: []byte(merge.Right), Token: []byte(merge.Token)}
	}
	data, err := json.Marshal(checkpointHeader{Config: m.Config, Tokens: tokens, Merges: merges, Values: valuesJSON})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return values
}

func TestLoadLegacyCheckpointRoundTrip(t *testing.T) {
	src, _ := smallModel(t)
	perturbForIntegrity(src)
	layout := legacyLayouts()[0]
	dir := t.TempDir()
	legacyPath := filepath.Join(dir, "legacy.json")
	writeLegacyCheckpoint(t, legacyPath, src, layout)

	// The current loader refuses it, and names the migration command.
	if _, err := LoadCheckpoint(legacyPath); err == nil {
		t.Fatal("current loader accepted a legacy checkpoint")
	} else if !strings.Contains(err.Error(), "migrate") || !strings.Contains(err.Error(), layout.Name) {
		t.Fatalf("rejection does not explain the migration: %v", err)
	}

	model, report, err := LoadLegacyCheckpoint(legacyPath)
	if err != nil {
		t.Fatal(err)
	}
	if report.Layout != layout.Name {
		t.Fatalf("layout = %q, want %q", report.Layout, layout.Name)
	}
	if want := src.Config.Context * src.Config.Width; report.DroppedPositionals != want {
		t.Fatalf("dropped positional values = %d, want %d", report.DroppedPositionals, want)
	}
	if !report.FeedForwardConverted || report.TargetParameters != model.ParameterCount() {
		t.Fatalf("report = %+v", report)
	}
	if !equalF32(model.tokenEmb.values, src.tokenEmb.values) {
		t.Fatal("token embedding not carried over")
	}

	// The migrated model is a valid current checkpoint that reloads identically
	// and evaluates to a finite loss.
	out := filepath.Join(dir, "migrated.json")
	if err := model.SaveCheckpoint(out); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCheckpoint(out)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Fingerprint() != model.Fingerprint() {
		t.Fatal("migrated checkpoint does not reload with the same fingerprint")
	}
	loss, err := loaded.EvaluateFast([]string{"alpha beta gamma delta"})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		t.Fatalf("migrated model produced loss %v", loss)
	}
}
