package lm

// Regression tests for checkpoint integrity (issue #59): corruption,
// truncation, incompatible configuration, atomic saves, and evaluation
// reproducibility/identity.

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// perturb makes the model's state non-default so round trips are meaningful.
func perturbForIntegrity(m *Model) {
	for i, p := range m.parameters {
		for j := range p.values {
			p.values[j] += float32(i*101+j) * 1e-6
			p.first[j] = float32(j%7) * 0.01
			p.second[j] = float32(j%5) * 0.02
		}
	}
	m.step = 1234
	m.seed = 77
}

// patchSafetensorsMeta rewrites a container's metadata map while leaving the
// tensor data untouched, simulating an edited or corrupted checkpoint.
func patchSafetensorsMeta(t *testing.T, path string, mutate func(map[string]string)) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	file, err := parseSafetensors(data)
	if err != nil {
		t.Fatalf("parse saved checkpoint: %v", err)
	}
	meta := file.metadata()
	mutate(meta)
	metaJSON, err := json.Marshal(meta)
	if err != nil {
		t.Fatal(err)
	}
	header := make(map[string]json.RawMessage, len(file.header))
	for name, raw := range file.header {
		header[name] = raw
	}
	header["__metadata__"] = metaJSON
	headerJSON, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]byte, 8, 8+len(headerJSON)+len(data)-file.base)
	binary.LittleEndian.PutUint64(out, uint64(len(headerJSON)))
	out = append(out, headerJSON...)
	out = append(out, data[file.base:]...)
	if err := os.WriteFile(path, out, 0o600); err != nil {
		t.Fatal(err)
	}
}

// patchSafetensorsFloat rewrites one float of a named tensor.
func patchSafetensorsFloat(t *testing.T, path, tensor string, index int, value float32) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	file, err := parseSafetensors(data)
	if err != nil {
		t.Fatalf("parse saved checkpoint: %v", err)
	}
	var entry stHeaderEntry
	if err := json.Unmarshal(file.header[tensor], &entry); err != nil {
		t.Fatalf("tensor %q header: %v", tensor, err)
	}
	offset := file.base + entry.DataOffsets[0] + index*4
	if offset+4 > len(data) {
		t.Fatalf("tensor %q index %d out of range", tensor, index)
	}
	binary.LittleEndian.PutUint32(data[offset:], math.Float32bits(value))
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestSaveRefusesInvalidModelState(t *testing.T) {
	tests := []struct {
		name    string
		corrupt func(*Model)
		want    string
	}{
		{name: "nan weight", corrupt: func(m *Model) { m.parameters[0].values[3] = float32(math.NaN()) }, want: "finite"},
		{name: "infinite weight", corrupt: func(m *Model) { m.parameters[1].values[0] = float32(math.Inf(1)) }, want: "finite"},
		{name: "nan first moment", corrupt: func(m *Model) { m.parameters[0].first[2] = float32(math.NaN()) }, want: "finite"},
		{name: "negative second moment", corrupt: func(m *Model) { m.parameters[0].second[0] = -0.5 }, want: "cannot be negative"},
		{name: "negative step", corrupt: func(m *Model) { m.step = -5 }, want: "negative"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			model, _ := smallModel(t)
			tc.corrupt(model)
			dir := t.TempDir()
			for _, path := range []string{
				filepath.Join(dir, "ckpt.json"),
				filepath.Join(dir, "ckpt.safetensors"),
			} {
				err := model.SaveCheckpoint(path)
				if err == nil {
					t.Fatalf("%s: save accepted invalid state", path)
				}
				if !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("%s: error %q does not mention %q", path, err, tc.want)
				}
				if _, statErr := os.Stat(path); statErr == nil {
					t.Fatalf("%s was written despite invalid state", path)
				}
			}
			if err := model.SaveSafetensors(filepath.Join(dir, "w.safetensors")); err == nil {
				t.Fatal("weights-only save accepted invalid state")
			}
		})
	}
}

func TestLoadRejectsNonFiniteWeights(t *testing.T) {
	model, _ := smallModel(t)
	path := filepath.Join(t.TempDir(), "ckpt.safetensors")
	if err := model.SaveSafetensorsCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	patchSafetensorsFloat(t, path, "token_emb", 1, float32(math.NaN()))
	if _, err := LoadCheckpoint(path); err == nil || !strings.Contains(err.Error(), "finite") {
		t.Fatalf("expected non-finite rejection, got %v", err)
	}
}

func TestLoadRejectsNonFiniteMoments(t *testing.T) {
	model, _ := smallModel(t)
	perturbForIntegrity(model)
	path := filepath.Join(t.TempDir(), "ckpt.safetensors")
	if err := model.SaveSafetensorsCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	patchSafetensorsFloat(t, path, "optimizer.first.token_emb", 0, float32(math.Inf(-1)))
	_, err := LoadCheckpoint(path)
	if err == nil || !strings.Contains(err.Error(), "optimizer.first.token_emb") {
		t.Fatalf("expected optimizer moment rejection naming the tensor, got %v", err)
	}
}

func TestLoadRejectsNegativeStep(t *testing.T) {
	model, _ := smallModel(t)
	perturbForIntegrity(model)
	dir := t.TempDir()

	jsonPath := filepath.Join(dir, "ckpt.json")
	if err := model.saveJSONCheckpoint(jsonPath); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatal(err)
	}
	patched := strings.Replace(string(data), `"step":1234`, `"step":-9`, 1)
	if err := os.WriteFile(jsonPath, []byte(patched), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(jsonPath); err == nil || !strings.Contains(err.Error(), "negative") {
		t.Fatalf("JSON: expected negative-step rejection, got %v", err)
	}

	stPath := filepath.Join(dir, "ckpt.safetensors")
	if err := model.SaveSafetensorsCheckpoint(stPath); err != nil {
		t.Fatal(err)
	}
	patchSafetensorsMeta(t, stPath, func(meta map[string]string) { meta["step"] = "-9" })
	if _, err := LoadCheckpoint(stPath); err == nil || !strings.Contains(err.Error(), "negative") {
		t.Fatalf("safetensors: expected negative-step rejection, got %v", err)
	}
}

func TestLoadRejectsInvalidConfig(t *testing.T) {
	model, _ := smallModel(t)
	dir := t.TempDir()

	jsonPath := filepath.Join(dir, "ckpt.json")
	if err := model.saveJSONCheckpoint(jsonPath); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatal(err)
	}
	patched := strings.Replace(string(data), `"width":8`, `"width":7`, 1)
	if err := os.WriteFile(jsonPath, []byte(patched), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(jsonPath); err == nil || !strings.Contains(err.Error(), "divisible") {
		t.Fatalf("JSON: expected config rejection, got %v", err)
	}

	stPath := filepath.Join(dir, "ckpt.safetensors")
	if err := model.SaveSafetensorsCheckpoint(stPath); err != nil {
		t.Fatal(err)
	}
	patchSafetensorsMeta(t, stPath, func(meta map[string]string) {
		meta["model_config"] = `{"context":8,"width":7,"layers":1,"heads":2}`
	})
	if _, err := LoadCheckpoint(stPath); err == nil || !strings.Contains(err.Error(), "divisible") {
		t.Fatalf("safetensors: expected config rejection, got %v", err)
	}

	// Negative layers must not be silently defaulted.
	patchSafetensorsMeta(t, stPath, func(meta map[string]string) {
		meta["model_config"] = `{"context":8,"width":8,"layers":-1,"heads":2}`
	})
	if _, err := LoadCheckpoint(stPath); err == nil || !strings.Contains(err.Error(), "layers") {
		t.Fatalf("safetensors: expected negative-layers rejection, got %v", err)
	}
}

func TestLoadRejectsMalformedTokenizer(t *testing.T) {
	model, _ := smallModel(t)
	perturbForIntegrity(model)
	path := filepath.Join(t.TempDir(), "ckpt.json")
	if err := model.saveJSONCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	// reset rewrites a pristine checkpoint and decodes it as a generic map, so
	// each corruption case starts from valid state.
	reset := func(t *testing.T) map[string]any {
		t.Helper()
		if err := model.saveJSONCheckpoint(path); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var doc map[string]any
		if err := json.Unmarshal(data, &doc); err != nil {
			t.Fatal(err)
		}
		return doc
	}
	write := func(t *testing.T, doc map[string]any) {
		t.Helper()
		data, err := json.Marshal(doc)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}

	// Duplicate token strings. The stored fingerprint is dropped so the
	// tokenizer-state check is what rejects the file, not the hash mismatch.
	doc := reset(t)
	tokens := doc["tokens"].([]any)
	tokens[1] = tokens[0]
	delete(doc, "tokenizer_hash")
	write(t, doc)
	if _, err := LoadCheckpoint(path); err == nil || !strings.Contains(err.Error(), "duplicates") {
		t.Fatalf("expected duplicate-token rejection, got %v", err)
	}

	// Special token id outside the vocabulary.
	doc = reset(t)
	doc["special_tokens"] = map[string]any{"<|pad|>": len(model.Tokenizer.Tokens) + 5}
	write(t, doc)
	if _, err := LoadCheckpoint(path); err == nil || !strings.Contains(err.Error(), "outside") {
		t.Fatalf("expected out-of-range special token rejection, got %v", err)
	}
}

func TestLoadRejectsTruncationAndTrailingData(t *testing.T) {
	model, _ := smallModel(t)
	perturbForIntegrity(model)
	dir := t.TempDir()

	stPath := filepath.Join(dir, "ckpt.safetensors")
	if err := model.SaveSafetensorsCheckpoint(stPath); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(stPath)
	if err != nil {
		t.Fatal(err)
	}

	truncated := filepath.Join(dir, "truncated.safetensors")
	if err := os.WriteFile(truncated, data[:len(data)-16], 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(truncated); err == nil {
		t.Fatal("expected truncation rejection")
	}

	trailing := filepath.Join(dir, "trailing.safetensors")
	if err := os.WriteFile(trailing, append(append([]byte{}, data...), 0, 0, 0, 0), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(trailing); err == nil || !strings.Contains(err.Error(), "trailing") {
		t.Fatalf("expected trailing-data rejection, got %v", err)
	}

	jsonPath := filepath.Join(dir, "ckpt.json")
	if err := model.saveJSONCheckpoint(jsonPath); err != nil {
		t.Fatal(err)
	}
	jsonData, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatal(err)
	}
	half := filepath.Join(dir, "half.json")
	if err := os.WriteFile(half, jsonData[:len(jsonData)/2], 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(half); err == nil {
		t.Fatal("expected truncated JSON rejection")
	}
}

func TestLoadRejectsInconsistentOptimizer(t *testing.T) {
	model, _ := smallModel(t)
	model.ResetOptimizer() // no moment tensors are written
	path := filepath.Join(t.TempDir(), "ckpt.safetensors")
	if err := model.SaveSafetensorsCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	patchSafetensorsMeta(t, path, func(meta map[string]string) { meta["has_optimizer"] = "1" })
	if _, err := LoadCheckpoint(path); err == nil || !strings.Contains(err.Error(), "missing tensor") {
		t.Fatalf("expected missing optimizer tensor rejection, got %v", err)
	}

	// A bogus has_optimizer value is rejected outright.
	patchSafetensorsMeta(t, path, func(meta map[string]string) { meta["has_optimizer"] = "yes" })
	if _, err := LoadCheckpoint(path); err == nil || !strings.Contains(err.Error(), "has_optimizer") {
		t.Fatalf("expected has_optimizer rejection, got %v", err)
	}
}

func TestLoadRejectsUnsupportedVersionAndMismatchedParameterCount(t *testing.T) {
	model, _ := smallModel(t)
	dir := t.TempDir()

	jsonPath := filepath.Join(dir, "ckpt.json")
	if err := model.saveJSONCheckpoint(jsonPath); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatal(err)
	}
	patched := strings.Replace(string(data), `"format_version":2`, `"format_version":9`, 1)
	if err := os.WriteFile(jsonPath, []byte(patched), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(jsonPath); err == nil || !strings.Contains(err.Error(), "format version") {
		t.Fatalf("expected version rejection, got %v", err)
	}

	stPath := filepath.Join(dir, "ckpt.safetensors")
	if err := model.SaveSafetensorsCheckpoint(stPath); err != nil {
		t.Fatal(err)
	}
	patchSafetensorsMeta(t, stPath, func(meta map[string]string) { meta["format_version"] = "2" })
	if _, err := LoadCheckpoint(stPath); err == nil || !strings.Contains(err.Error(), "format version") {
		t.Fatalf("safetensors: expected version rejection, got %v", err)
	}
	patchSafetensorsMeta(t, stPath, func(meta map[string]string) {
		meta["format_version"] = "1"
		meta["parameter_count"] = "3"
	})
	if _, err := LoadCheckpoint(stPath); err == nil || !strings.Contains(err.Error(), "parameter_count") {
		t.Fatalf("expected parameter_count rejection, got %v", err)
	}
}

func TestWriteFileAtomicKeepsPreviousCheckpoint(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "state.json")
	if err := os.WriteFile(path, []byte("previous"), 0o600); err != nil {
		t.Fatal(err)
	}

	sentinel := errors.New("writer failed")
	err := writeFileAtomic(path, 0o600, func(w io.Writer) error {
		if _, err := io.WriteString(w, "partial"); err != nil {
			return err
		}
		return sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want the writer error", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "previous" {
		t.Fatalf("failed write clobbered the checkpoint: %q", got)
	}
	assertOnlyFile(t, dir, "state.json")

	if err := writeFileAtomic(path, 0o600, func(w io.Writer) error {
		_, err := io.WriteString(w, "replacement")
		return err
	}); err != nil {
		t.Fatal(err)
	}
	got, err = os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "replacement" {
		t.Fatalf("content = %q", got)
	}
	assertOnlyFile(t, dir, "state.json")

	// The destination directory is created on demand.
	if err := writeFileAtomic(filepath.Join(dir, "nested", "ckpt.bin"), 0o600, func(w io.Writer) error {
		_, err := io.WriteString(w, "nested")
		return err
	}); err != nil {
		t.Fatal(err)
	}
}

// assertOnlyFile fails when dir holds anything other than one entry named name,
// which is how a leaked temporary file would show up.
func assertOnlyFile(t *testing.T, dir, name string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != name {
		names := make([]string, len(entries))
		for i, e := range entries {
			names[i] = e.Name()
		}
		t.Fatalf("directory holds %v, want only %q", names, name)
	}
}

func TestFingerprintIsContainerIndependent(t *testing.T) {
	model, _ := smallModel(t)
	perturbForIntegrity(model)
	model.ResetOptimizer()
	dir := t.TempDir()

	jsonPath := filepath.Join(dir, "ckpt.json")
	if err := model.saveJSONCheckpoint(jsonPath); err != nil {
		t.Fatal(err)
	}
	stPath := filepath.Join(dir, "ckpt.safetensors")
	if err := model.SaveSafetensorsCheckpoint(stPath); err != nil {
		t.Fatal(err)
	}
	fromJSON, err := LoadCheckpoint(jsonPath)
	if err != nil {
		t.Fatal(err)
	}
	fromST, err := LoadCheckpoint(stPath)
	if err != nil {
		t.Fatal(err)
	}
	want := model.Fingerprint()
	if got := fromJSON.Fingerprint(); got != want {
		t.Fatalf("JSON fingerprint %s != %s", got, want)
	}
	if got := fromST.Fingerprint(); got != want {
		t.Fatalf("safetensors fingerprint %s != %s", got, want)
	}

	fromST.parameters[0].values[0] += 1
	if fromST.Fingerprint() == want {
		t.Fatal("fingerprint did not change when a weight changed")
	}
}

func TestEvaluationIsReproducibleAcrossLoads(t *testing.T) {
	model, _ := smallModel(t)
	perturbForIntegrity(model)
	texts := []string{"alpha beta gamma delta. epsilon zeta theta.", "more text for evaluation"}

	first, err := model.EvaluateFast(texts)
	if err != nil {
		t.Fatal(err)
	}
	again, err := model.EvaluateFast(texts)
	if err != nil {
		t.Fatal(err)
	}
	if first != again {
		t.Fatalf("repeated evaluation differs: %v vs %v", first, again)
	}

	path := filepath.Join(t.TempDir(), "ckpt.json")
	if err := model.SaveCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	reloaded, err := loaded.EvaluateFast(texts)
	if err != nil {
		t.Fatal(err)
	}
	if reloaded != first {
		t.Fatalf("evaluation after reload differs: %v vs %v", reloaded, first)
	}

	// Identities of the same weights agree.
	idA, err := BuildEvalIdentity(model, EvalIdentityInput{
		Mode: EvalModeFast, Backend: "cpu", DatasetTexts: texts, TokenCount: 12,
	})
	if err != nil {
		t.Fatal(err)
	}
	idB, err := BuildEvalIdentity(loaded, EvalIdentityInput{
		Mode: EvalModeFast, Backend: "cpu", DatasetTexts: texts, TokenCount: 12,
	})
	if err != nil {
		t.Fatal(err)
	}
	if idA.ModelSHA256 != idB.ModelSHA256 {
		t.Fatalf("model fingerprints differ across loads: %s vs %s", idA.ModelSHA256, idB.ModelSHA256)
	}
	if cmp := CompareEvalIdentities(idA, idB); !cmp.Compatible || len(cmp.Refusals) != 0 {
		t.Fatalf("identical evaluations refused: %+v", cmp)
	}
}

func TestCompareEvalIdentities(t *testing.T) {
	base := EvalIdentity{
		EvaluatorVersion: EvaluatorVersion,
		EvalMode:         EvalModeFast,
		Backend:          "cpu",
		CheckpointSHA256: "aaaa",
		ModelSHA256:      "model",
		TokenizerHash:    "fnv64a:1",
		Config:           Config{Context: 64, Width: 32, Layers: 2, Heads: 4},
		Context:          64,
		DatasetSHA256:    "data",
		DatasetSplit:     "held-out",
		TokenCount:       100,
	}
	with := func(mutate func(*EvalIdentity)) EvalIdentity {
		out := base
		mutate(&out)
		return out
	}

	tests := []struct {
		name        string
		current     EvalIdentity
		wantRefusal string
		wantWarning string
	}{
		{name: "same evaluation, different checkpoint", current: with(func(e *EvalIdentity) { e.CheckpointSHA256 = "bbbb" })},
		{name: "evaluator version", current: with(func(e *EvalIdentity) { e.EvaluatorVersion = "eval/2" }), wantRefusal: "evaluator version"},
		{name: "eval mode", current: with(func(e *EvalIdentity) { e.EvalMode = EvalModeExact }), wantRefusal: "eval mode"},
		{name: "context", current: with(func(e *EvalIdentity) { e.Context = 128 }), wantRefusal: "context window"},
		{name: "config", current: with(func(e *EvalIdentity) {
			e.Config.Width = 64
			e.Context = 64
		}), wantRefusal: "model config"},
		{name: "tokenizer", current: with(func(e *EvalIdentity) { e.TokenizerHash = "fnv64a:2" }), wantRefusal: "tokenizer"},
		{name: "dataset", current: with(func(e *EvalIdentity) { e.DatasetSHA256 = "other" }), wantRefusal: "dataset fingerprint"},
		{name: "token count", current: with(func(e *EvalIdentity) { e.TokenCount = 99 }), wantRefusal: "evaluated token count"},
		{name: "backend differs", current: with(func(e *EvalIdentity) { e.Backend = "cuda" }), wantWarning: "backend"},
		{name: "split differs", current: with(func(e *EvalIdentity) { e.DatasetSplit = "val" }), wantWarning: "dataset split"},
		{name: "same checkpoint", current: base, wantWarning: "same checkpoint"},
		{name: "legacy baseline", current: base, wantWarning: "predates"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			baseline := base
			if tc.name == "legacy baseline" {
				baseline = EvalIdentity{}
			}
			cmp := CompareEvalIdentities(baseline, tc.current)
			if tc.wantRefusal != "" {
				if cmp.Compatible {
					t.Fatalf("expected refusal, got compatible: %+v", cmp)
				}
				if !containsSubstring(cmp.Refusals, tc.wantRefusal) {
					t.Fatalf("refusals %v do not mention %q", cmp.Refusals, tc.wantRefusal)
				}
				return
			}
			if !cmp.Compatible {
				t.Fatalf("expected compatibility, got refusals %v", cmp.Refusals)
			}
			if tc.wantWarning != "" && !containsSubstring(cmp.Warnings, tc.wantWarning) {
				t.Fatalf("warnings %v do not mention %q", cmp.Warnings, tc.wantWarning)
			}
		})
	}
}

func containsSubstring(haystack []string, needle string) bool {
	for _, s := range haystack {
		if strings.Contains(s, needle) {
			return true
		}
	}
	return false
}

func TestEvalIdentityJSONRoundTrip(t *testing.T) {
	model, _ := smallModel(t)
	id, err := BuildEvalIdentity(model, EvalIdentityInput{
		Mode:           EvalModeExact,
		Backend:        "cpu",
		CheckpointPath: "",
		DatasetTexts:   []string{"alpha beta"},
		DatasetSplit:   "held-out",
		DatasetItems:   1,
		DatasetTokens:  10,
		TokenCount:     9,
	})
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(id)
	if err != nil {
		t.Fatal(err)
	}
	var back EvalIdentity
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if back != id {
		t.Fatalf("identity did not round trip:\n%+v\n%+v", id, back)
	}
	if id.EvaluatorVersion != EvaluatorVersion || id.Context != model.Config.Context {
		t.Fatalf("identity missing evaluator/context: %+v", id)
	}
	if id.DatasetSHA256 != textsSHA256([]string{"alpha beta"}) {
		t.Fatal("dataset fingerprint does not match the evaluated texts")
	}
}
