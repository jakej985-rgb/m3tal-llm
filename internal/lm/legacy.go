package lm

// Import of checkpoints written before the current architecture.
//
// Two historical layouts exist in this repository's history:
//
//   - v1.pos_emb.mlp (commits up to 734f874): a learned positional embedding
//     table, four attention projections, and a feed-forward block made of two
//     matrices with a tanh activation (ffIn -> tanh -> ffOut). No RMSNorm.
//   - v1.rope.mlp (the uncommitted state between 734f874 and b18669c, which
//     several local checkpoints were trained with): RoPE instead of the
//     positional table, two RMSNorm gammas per block, and still the two-matrix
//     tanh MLP.
//
// The current architecture has no positional table (RoPE supplies position)
// and fuses the feed-forward block into SwiGLU (gate/up/down). A legacy file
// therefore does not match the current parameter count, and LoadCheckpoint
// rejects it. This file detects those layouts and maps them onto the current
// model so an old checkpoint can be re-exported and fine-tuned.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
)

// Legacy layout identifiers recorded in import reports.
const (
	LegacyLayoutPosEmbMLP = "v1.pos_emb.mlp"
	LegacyLayoutRopeMLP   = "v1.rope.mlp"
)

// LegacyLayout identifies an importable pre-current parameter layout.
type LegacyLayout struct {
	Name string
	// PositionalEmbeddings is true when the checkpoint carries a learned
	// [context,width] table that the current architecture has no place for.
	PositionalEmbeddings bool
	// NormGammas is true when each block stores two RMSNorm gamma vectors.
	// Without them the current gammas keep their identity initialization.
	NormGammas bool
}

func legacyLayouts() []LegacyLayout {
	return []LegacyLayout{
		{Name: LegacyLayoutPosEmbMLP, PositionalEmbeddings: true, NormGammas: false},
		{Name: LegacyLayoutRopeMLP, PositionalEmbeddings: false, NormGammas: true},
	}
}

// legacyParameterCount returns the number of values a legacy container holds.
// Per block: four attention linears (4*(w*w+w)) plus the two-matrix MLP
// (ffIn w->2w and ffOut 2w->w = 6*w*w+3*w) plus two gammas when present.
func legacyParameterCount(cfg Config, vocab int, layout LegacyLayout) int {
	cfg.Normalize()
	w, layers := cfg.Width, cfg.Layers
	total := vocab*w + w*vocab + vocab
	if layout.PositionalEmbeddings {
		total += cfg.Context * w
	}
	perBlock := 4*(w*w+w) + (2*w*w + 2*w) + (2*w*w + w)
	if layout.NormGammas {
		perBlock += 2 * w
	}
	return total + layers*perBlock
}

// DetectLegacyLayout reports which legacy layout matches a parameter count for
// this config and vocabulary.
func DetectLegacyLayout(cfg Config, vocab, count int) (LegacyLayout, bool) {
	for _, layout := range legacyLayouts() {
		if legacyParameterCount(cfg, vocab, layout) == count {
			return layout, true
		}
	}
	return LegacyLayout{}, false
}

// LegacyImportReport describes what an import changed.
type LegacyImportReport struct {
	Layout                string `json:"layout"`
	SourceParameters      int    `json:"source_parameters"`
	TargetParameters      int    `json:"target_parameters"`
	DroppedPositionals    int    `json:"dropped_positional_values"`
	FeedForwardConverted  bool   `json:"feed_forward_converted"`
	NormGammasInitialized bool   `json:"norm_gammas_initialized"`
	OptimizerDropped      bool   `json:"optimizer_state_dropped"`
}

// ImportLegacy copies flat legacy parameters into m:
//
//   - token embedding, attention projections and the output projection are
//     copied unchanged;
//   - the learned positional embedding table is dropped, because the current
//     architecture encodes position with RoPE;
//   - blocks without stored RMSNorm gammas keep the identity initialization
//     (gamma = 1). The legacy blocks had no normalization at all, so this is
//     the closest available setting: RMSNorm still rescales each sublayer
//     input to unit RMS, but adds no learnable scaling on top;
//   - the two-matrix tanh MLP becomes the SwiGLU triple with gate = ffIn,
//     up = 1 (zero weights, unit bias) and down = ffOut. Both linear maps are
//     preserved exactly; only the activation changes (silu instead of tanh).
//   - optimizer moments are reset, since they belong to the old graph.
//
// The result is a fine-tuning starting point, not a numerically equivalent
// model: positional information and the activation differ.
func (m *Model) ImportLegacy(layout LegacyLayout, flat []float32) error {
	m.Config.Normalize()
	vocab, width, context := len(m.Tokenizer.Tokens), m.Config.Width, m.Config.Context
	want := legacyParameterCount(m.Config, vocab, layout)
	if len(flat) != want {
		return fmt.Errorf("legacy import: %d values do not match the %s layout (%d values)",
			len(flat), layout.Name, want)
	}
	if len(m.blocks) != m.Config.Layers {
		return fmt.Errorf("legacy import: model has %d blocks, config says %d", len(m.blocks), m.Config.Layers)
	}

	offset := 0
	take := func(n int) []float32 {
		part := flat[offset : offset+n]
		offset += n
		return part
	}
	copyLinear := func(l linear) {
		copy(l.weights.values, take(l.input*l.output))
		copy(l.bias.values, take(l.output))
	}

	copy(m.tokenEmb.values, take(vocab*width))
	if layout.PositionalEmbeddings {
		_ = take(context * width) // dropped: RoPE supplies position now
	}

	hidden := width * 2
	for i := range m.blocks {
		b := &m.blocks[i]
		if layout.NormGammas {
			copy(b.normAttn.values, take(width))
			copy(b.normFF.values, take(width))
		} else {
			for k := range b.normAttn.values {
				b.normAttn.values[k] = 1
			}
			for k := range b.normFF.values {
				b.normFF.values[k] = 1
			}
		}
		copyLinear(b.query)
		copyLinear(b.key)
		copyLinear(b.value)
		copyLinear(b.attnOut)

		// Legacy MLP: ffIn [width,hidden]+bias, ffOut [hidden,width]+bias.
		ffInWeights := take(width * hidden)
		ffInBias := take(hidden)
		ffOutWeights := take(hidden * width)
		ffOutBias := take(width)
		copy(b.ffGate.weights.values, ffInWeights)
		copy(b.ffGate.bias.values, ffInBias)
		clear(b.ffUp.weights.values)
		for k := range b.ffUp.bias.values {
			b.ffUp.bias.values[k] = 1
		}
		copy(b.ffDown.weights.values, ffOutWeights)
		copy(b.ffDown.bias.values, ffOutBias)
	}
	copyLinear(m.output)

	if offset != len(flat) {
		return fmt.Errorf("legacy import: consumed %d of %d values", offset, len(flat))
	}
	m.step = 0
	m.ResetOptimizer()
	m.invalidateBackend()
	return nil
}

// LoadLegacyCheckpoint reads a pre-current-architecture JSON checkpoint,
// detects its layout, and builds an equivalent current model. The returned
// report describes the conversion. Optimizer state in the source file is
// ignored.
func LoadLegacyCheckpoint(path string) (*Model, LegacyImportReport, error) {
	var report LegacyImportReport
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, report, err
	}
	if isSafetensorsContainer(data) {
		return nil, report, fmt.Errorf("%s: safetensors containers are always written by the current architecture; nothing to migrate", path)
	}
	decoded, err := decodeCheckpoint(data)
	if err != nil {
		return nil, report, fmt.Errorf("decode legacy checkpoint: %w", err)
	}
	header := decoded.header
	tokenizer := tokenizerFromCheckpoint(header.Tokens, header.Merges, header.SpecialTokens)
	if err := tokenizer.Validate(); err != nil {
		return nil, report, err
	}
	flat, err := parseFloatArray(decoded.values)
	if err != nil {
		return nil, report, fmt.Errorf("legacy checkpoint values: %w", err)
	}
	layout, ok := DetectLegacyLayout(header.Config, len(tokenizer.Tokens), len(flat))
	if !ok {
		model, err := NewModel(header.Config, tokenizer, 1)
		if err != nil {
			return nil, report, err
		}
		return nil, report, fmt.Errorf("%s: %d parameters match no legacy layout for this config (current layout needs %d)",
			path, len(flat), model.ParameterCount())
	}

	model, err := NewModel(header.Config, tokenizer, 1)
	if err != nil {
		return nil, report, err
	}
	if err := model.ImportLegacy(layout, flat); err != nil {
		return nil, report, err
	}
	if header.Seed != 0 {
		model.seed = header.Seed
	}
	if err := model.validateModelState(); err != nil {
		return nil, report, fmt.Errorf("legacy checkpoint integrity: %w", err)
	}
	report = LegacyImportReport{
		Layout:                layout.Name,
		SourceParameters:      len(flat),
		TargetParameters:      model.ParameterCount(),
		DroppedPositionals:    contextPositionalValues(header.Config, layout),
		FeedForwardConverted:  true,
		NormGammasInitialized: !layout.NormGammas,
		OptimizerDropped:      decoded.hasOptimizer,
	}
	return model, report, nil
}

// contextPositionalValues reports how many positional-embedding values an
// import discards.
func contextPositionalValues(cfg Config, layout LegacyLayout) int {
	if !layout.PositionalEmbeddings {
		return 0
	}
	cfg.Normalize()
	return cfg.Context * cfg.Width
}

// parseFloatArray decodes a JSON number array of any length.
func parseFloatArray(raw []byte) ([]float32, error) {
	n := countJSONNumbers(raw)
	out := make([]float32, n)
	if err := parseFloatSliceInto(raw, out); err != nil {
		return nil, err
	}
	return out, nil
}

// countJSONNumbers returns how many numbers a JSON array literal holds, so a
// legacy array can be sized before parsing. A malformed array yields a count
// that makes the parser report the error.
func countJSONNumbers(raw []byte) int {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) < 2 || trimmed[0] != '[' || trimmed[len(trimmed)-1] != ']' {
		return 0
	}
	body := bytes.TrimSpace(trimmed[1 : len(trimmed)-1])
	if len(body) == 0 {
		return 0
	}
	return bytes.Count(body, []byte(",")) + 1
}

// legacyLayoutHint explains an unsupported parameter count when the file looks
// like a legacy checkpoint, and returns "" when it does not.
func legacyLayoutHint(cfg Config, vocab, count int) string {
	layout, ok := DetectLegacyLayout(cfg, vocab, count)
	if !ok {
		return ""
	}
	return fmt.Sprintf("this looks like a legacy %s checkpoint (%d values): re-export it with `m3tal-lm migrate --in <file> --out <new-file>`",
		layout.Name, count)
}

// legacyNote renders a one-line human summary of an import report.
func (r LegacyImportReport) Note() string {
	note := fmt.Sprintf("layout %s: %d -> %d parameters", r.Layout, r.SourceParameters, r.TargetParameters)
	if r.DroppedPositionals > 0 {
		note += fmt.Sprintf("; dropped %d learned positional values (RoPE now)", r.DroppedPositionals)
	}
	if r.NormGammasInitialized {
		note += "; RMSNorm gammas initialized to 1 (legacy blocks were unnormalized)"
	}
	if r.FeedForwardConverted {
		note += "; tanh MLP mapped to SwiGLU (gate=ffIn, up=1, down=ffOut), so the activation differs until fine-tuning"
	}
	if r.OptimizerDropped {
		note += "; optimizer state dropped"
	}
	return note
}

// MarshalLegacyValuesJSON renders values as a JSON array, used by tests and by
// re-export tooling.
func MarshalLegacyValuesJSON(values []float32) (json.RawMessage, error) {
	data, err := json.Marshal(values)
	if err != nil {
		return nil, err
	}
	return json.RawMessage(data), nil
}
