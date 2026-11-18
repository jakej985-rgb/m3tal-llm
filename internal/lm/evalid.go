package lm

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

// This file defines evaluation identity: the immutable facts a reported metric
// depends on. Every eval report records them, and comparisons refuse or warn
// when two evaluations measured different things.

// EvaluatorVersion identifies the scoring semantics behind an evaluation.
// Bump it whenever loss computation, prompt construction, truncation or
// decoding defaults change, so reports from different evaluators are never
// compared as if they measured the same quantity.
const EvaluatorVersion = "eval/1"

// Recorded evaluation modes.
const (
	EvalModeFast  = "fast"
	EvalModeExact = "exact"
	EvalModeChat  = "chat"
)

// EvalIdentity is the identity an evaluation result belongs to: the model, the
// tokenizer, the data, the context window, the evaluation mode and the
// backend. Hashes are lowercase hex sha256 unless noted.
type EvalIdentity struct {
	EvaluatorVersion string `json:"evaluator_version"`
	EvalMode         string `json:"eval_mode"`
	Backend          string `json:"backend"`
	CheckpointSHA256 string `json:"checkpoint_sha256,omitempty"`
	ModelSHA256      string `json:"model_sha256"`
	TokenizerHash    string `json:"tokenizer_hash"`
	Config           Config `json:"model_config"`
	Context          int    `json:"context"`
	DatasetSHA256    string `json:"dataset_sha256"`
	DatasetSplit     string `json:"dataset_split,omitempty"`
	DatasetItems     int    `json:"dataset_items,omitempty"`
	DatasetTokens    int64  `json:"dataset_tokens,omitempty"`
	TokenCount       int64  `json:"eval_tokens"`
}

// EvalIdentityInput collects the run-specific inputs needed to build an
// identity. When DatasetPaths is set the data is hashed from the files on
// disk; otherwise DatasetTexts is hashed, so in-memory corpora are bound just
// as tightly.
type EvalIdentityInput struct {
	Mode           string
	Backend        string
	CheckpointPath string
	DatasetPaths   []string
	DatasetTexts   []string
	DatasetSplit   string
	DatasetItems   int
	DatasetTokens  int64
	TokenCount     int64
}

// BuildEvalIdentity computes the identity for one evaluation run.
func BuildEvalIdentity(m *Model, in EvalIdentityInput) (EvalIdentity, error) {
	if m == nil {
		return EvalIdentity{}, fmt.Errorf("eval identity: nil model")
	}
	m.Config.Normalize()
	id := EvalIdentity{
		EvaluatorVersion: EvaluatorVersion,
		EvalMode:         in.Mode,
		Backend:          in.Backend,
		ModelSHA256:      m.Fingerprint(),
		TokenizerHash:    TokenizerFingerprint(m.Tokenizer),
		Config:           m.Config,
		Context:          m.Config.Context,
		DatasetSplit:     in.DatasetSplit,
		DatasetItems:     in.DatasetItems,
		DatasetTokens:    in.DatasetTokens,
		TokenCount:       in.TokenCount,
	}
	if in.CheckpointPath != "" {
		hash, err := FileSHA256(in.CheckpointPath)
		if err != nil {
			return EvalIdentity{}, fmt.Errorf("eval identity: checkpoint hash: %w", err)
		}
		id.CheckpointSHA256 = hash
	}
	if len(in.DatasetPaths) > 0 {
		hash, err := filesSHA256(in.DatasetPaths)
		if err != nil {
			return EvalIdentity{}, fmt.Errorf("eval identity: dataset hash: %w", err)
		}
		id.DatasetSHA256 = hash
	} else {
		id.DatasetSHA256 = textsSHA256(in.DatasetTexts)
	}
	return id, nil
}

// EvalCompatibility is the outcome of comparing two evaluation identities.
type EvalCompatibility struct {
	Compatible bool     `json:"compatible"`
	Refusals   []string `json:"refusals,omitempty"`
	Warnings   []string `json:"warnings,omitempty"`
}

// CompareEvalIdentities decides whether two evaluations can be compared.
// Differences that change what is being measured (evaluator, mode, context,
// model shape, tokenizer, dataset, scored token count) are refusals: a delta
// between them is meaningless. Differences that only perturb numerics
// (backend) or are expected (a different checkpoint, a different split name)
// are warnings.
func CompareEvalIdentities(base, cur EvalIdentity) EvalCompatibility {
	out := EvalCompatibility{Compatible: true}
	if base.EvaluatorVersion == "" {
		out.Warnings = append(out.Warnings,
			fmt.Sprintf("baseline predates evaluation identity recording; comparison is unverified (current evaluator %s)", cur.EvaluatorVersion))
		return out
	}
	refuse := func(format string, args ...any) {
		out.Refusals = append(out.Refusals, fmt.Sprintf(format, args...))
	}
	warn := func(format string, args ...any) {
		out.Warnings = append(out.Warnings, fmt.Sprintf(format, args...))
	}

	if base.EvaluatorVersion != cur.EvaluatorVersion {
		refuse("evaluator version %s vs %s", base.EvaluatorVersion, cur.EvaluatorVersion)
	}
	if base.EvalMode != cur.EvalMode {
		refuse("eval mode %q vs %q", base.EvalMode, cur.EvalMode)
	}
	if base.Context != cur.Context {
		refuse("context window %d vs %d", base.Context, cur.Context)
	}
	if base.Config != cur.Config {
		refuse("model config %s vs %s", mustJSON(base.Config), mustJSON(cur.Config))
	}
	if base.TokenizerHash != cur.TokenizerHash {
		refuse("tokenizer %s vs %s", base.TokenizerHash, cur.TokenizerHash)
	}
	if base.DatasetSHA256 != cur.DatasetSHA256 {
		refuse("dataset fingerprint %s vs %s", base.DatasetSHA256, cur.DatasetSHA256)
	}
	if base.TokenCount != cur.TokenCount {
		refuse("evaluated token count %d vs %d", base.TokenCount, cur.TokenCount)
	}
	if base.DatasetSplit != cur.DatasetSplit {
		warn("dataset split %q vs %q", base.DatasetSplit, cur.DatasetSplit)
	}
	if base.Backend != cur.Backend {
		warn("backend %q vs %q: kernels reduce in a different order, so small loss differences are expected", base.Backend, cur.Backend)
	}
	if base.CheckpointSHA256 != "" && base.CheckpointSHA256 == cur.CheckpointSHA256 {
		warn("baseline used the same checkpoint (%s): the delta measures run-to-run noise, not a model change", cur.CheckpointSHA256)
	}
	out.Compatible = len(out.Refusals) == 0
	return out
}

// Fingerprint returns a sha256 over the model's identity: config, tokenizer
// fingerprint, and every weight bit in parameter order. Equal fingerprints mean
// bit-identical weights, regardless of which checkpoint container produced
// them. Optimizer state and step are deliberately excluded.
func (m *Model) Fingerprint() string {
	h := sha256.New()
	io.WriteString(h, "m3tal.model.v1\x00")
	cfg, err := json.Marshal(m.Config)
	if err != nil {
		// Config is four ints; marshalling cannot fail.
		fmt.Fprintf(h, "config-error:%v", err)
	}
	h.Write(cfg)
	io.WriteString(h, "\x00"+TokenizerFingerprint(m.Tokenizer)+"\x00")
	var buf [4]byte
	writeName := func(name string) {
		io.WriteString(h, name)
		h.Write([]byte{0})
	}
	for _, s := range m.parameterSpecs() {
		writeName(s.Name)
		for _, v := range s.P.values {
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
			h.Write(buf[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// FileSHA256 returns the sha256 of a file's bytes as lowercase hex.
func FileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// filesSHA256 hashes a list of corpus files in order, binding an evaluation to
// the exact bytes on disk.
func filesSHA256(paths []string) (string, error) {
	h := sha256.New()
	for _, path := range paths {
		f, err := os.Open(path)
		if err != nil {
			return "", err
		}
		io.WriteString(h, path)
		h.Write([]byte{0})
		if _, err := io.Copy(h, f); err != nil {
			f.Close()
			return "", err
		}
		f.Close()
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// textsSHA256 hashes in-memory evaluation texts with length prefixes so no
// concatenation of two inputs can collide with another.
func textsSHA256(texts []string) string {
	h := sha256.New()
	var lenBuf [8]byte
	io.WriteString(h, "m3tal.texts.v1\x00")
	for _, text := range texts {
		binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(text)))
		h.Write(lenBuf[:])
		io.WriteString(h, text)
	}
	return hex.EncodeToString(h.Sum(nil))
}
