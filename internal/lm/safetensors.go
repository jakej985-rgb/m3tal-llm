package lm

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
)

// This file implements read/write for the safetensors format
// (https://github.com/huggingface/safetensors): an 8-byte little-endian header
// length, a JSON header naming typed tensors and their byte offsets, then the
// tensor data.
//
// Two flavours are written:
//
//   - SaveSafetensors stores weights only, for interchange with other tools.
//   - SaveSafetensorsCheckpoint stores weights plus training state (config,
//     tokenizer, step, seed and Adam moments) so a run can resume. Large
//     SaveCheckpoint calls use it: the equivalent JSON is hundreds of
//     megabytes of text that has to be parsed number by number.
//
// LoadCheckpoint accepts both containers, plus the older JSON checkpoints.

const (
	safetensorsFormat           = "m3tal.safetensors"
	safetensorsCheckpointFormat = "m3tal.checkpoint.safetensors"
)

// parameterSpec names one trainable parameter, mirroring the naming used by
// gradcheck ParametersInfo.
type parameterSpec struct {
	Name string
	P    *parameter
}

func (m *Model) parameterSpecs() []parameterSpec {
	var specs []parameterSpec
	add := func(name string, p *parameter) {
		specs = append(specs, parameterSpec{Name: name, P: p})
	}
	addLinear := func(prefix string, l linear) {
		add(prefix+".weights", l.weights)
		add(prefix+".bias", l.bias)
	}
	add("token_emb", m.tokenEmb)
	for i, b := range m.blocks {
		prefix := fmt.Sprintf("blocks.%d", i)
		add(prefix+".norm_attn", b.normAttn)
		add(prefix+".norm_ff", b.normFF)
		addLinear(prefix+".query", b.query)
		addLinear(prefix+".key", b.key)
		addLinear(prefix+".value", b.value)
		addLinear(prefix+".attn_out", b.attnOut)
		addLinear(prefix+".ff_gate", b.ffGate)
		addLinear(prefix+".ff_up", b.ffUp)
		addLinear(prefix+".ff_down", b.ffDown)
	}
	addLinear("output", m.output)
	return specs
}

type stHeaderEntry struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets []int  `json:"data_offsets"`
}

type stTensor struct {
	name   string
	values []float32
}

// writeSafetensors writes the tensors and metadata map to path atomically.
func writeSafetensors(path string, tensors []stTensor, meta map[string]string) error {
	header := make(map[string]json.RawMessage, len(tensors)+1)
	metaJSON, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	header["__metadata__"] = metaJSON

	total := 0
	for _, t := range tensors {
		entry, err := json.Marshal(stHeaderEntry{
			Dtype:       "F32",
			Shape:       []int{len(t.values)},
			DataOffsets: []int{total, total + len(t.values)*4},
		})
		if err != nil {
			return err
		}
		header[t.name] = entry
		total += len(t.values) * 4
	}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return err
	}

	return writeFileAtomic(path, 0o600, func(w io.Writer) error {
		// Tensor data streams through one bounded scratch buffer: a large
		// checkpoint is hundreds of megabytes and must not need a second copy
		// in memory.
		var headerLen [8]byte
		binary.LittleEndian.PutUint64(headerLen[:], uint64(len(headerJSON)))
		if _, err := w.Write(headerLen[:]); err != nil {
			return err
		}
		if _, err := w.Write(headerJSON); err != nil {
			return err
		}
		scratch := make([]byte, 0, 1<<20)
		for _, t := range tensors {
			scratch = scratch[:0]
			for _, v := range t.values {
				scratch = binary.LittleEndian.AppendUint32(scratch, math.Float32bits(v))
				if len(scratch) == cap(scratch) {
					if _, err := w.Write(scratch); err != nil {
						return err
					}
					scratch = scratch[:0]
				}
			}
			if len(scratch) > 0 {
				if _, err := w.Write(scratch); err != nil {
					return err
				}
			}
		}
		return nil
	})
}

// SaveSafetensors writes all model weights to path in safetensors format.
// The file is written atomically (temporary file, fsync, rename) and refused
// when the model holds non-finite values.
func (m *Model) SaveSafetensors(path string) error {
	if err := m.validateModelState(); err != nil {
		return fmt.Errorf("refusing to save weights: %w", err)
	}
	specs := m.parameterSpecs()
	tensors := make([]stTensor, len(specs))
	for i, s := range specs {
		tensors[i] = stTensor{name: s.Name, values: s.P.values}
	}
	return writeSafetensors(path, tensors, map[string]string{
		"format":          safetensorsFormat,
		"format_version":  "1",
		"model_config":    mustJSON(m.Config),
		"tokenizer_hash":  TokenizerFingerprint(m.Tokenizer),
		"parameter_count": strconv.Itoa(len(specs)),
	})
}

// SaveSafetensorsCheckpoint writes weights and everything needed to resume
// training (config, tokenizer, step, seed, Adam moments) to one container.
func (m *Model) SaveSafetensorsCheckpoint(path string) error {
	if err := m.validateModelState(); err != nil {
		return fmt.Errorf("refusing to save checkpoint: %w", err)
	}
	specs := m.parameterSpecs()
	tensors := make([]stTensor, 0, len(specs)*3)
	for _, s := range specs {
		tensors = append(tensors, stTensor{name: s.Name, values: s.P.values})
	}
	hasOptimizer := m.hasOptimizerMoments()
	if hasOptimizer {
		for _, s := range specs {
			tensors = append(tensors, stTensor{name: "optimizer.first." + s.Name, values: s.P.first})
		}
		for _, s := range specs {
			tensors = append(tensors, stTensor{name: "optimizer.second." + s.Name, values: s.P.second})
		}
	}

	tokens := make([][]byte, len(m.Tokenizer.Tokens))
	for i, token := range m.Tokenizer.Tokens {
		tokens[i] = []byte(token)
	}
	merges := make([]checkpointMerge, len(m.Tokenizer.Merges))
	for i, merge := range m.Tokenizer.Merges {
		merges[i] = checkpointMerge{Left: []byte(merge.Left), Right: []byte(merge.Right), Token: []byte(merge.Token)}
	}
	tokenizer, err := json.Marshal(checkpointTokenizer{
		Tokens:        tokens,
		Merges:        merges,
		SpecialTokens: m.Tokenizer.SpecialTokens,
	})
	if err != nil {
		return err
	}

	hasOptimizerFlag := "0"
	if hasOptimizer {
		hasOptimizerFlag = "1"
	}
	return writeSafetensors(path, tensors, map[string]string{
		"format":          safetensorsCheckpointFormat,
		"format_version":  "1",
		"model_config":    mustJSON(m.Config),
		"step":            strconv.FormatInt(m.step, 10),
		"seed":            strconv.FormatInt(m.seed, 10),
		"tokenizer_hash":  TokenizerFingerprint(m.Tokenizer),
		"tokenizer":       string(tokenizer),
		"has_optimizer":   hasOptimizerFlag,
		"parameter_count": strconv.Itoa(len(specs)),
	})
}

func mustJSON(v interface{}) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// safetensorsFile is a parsed container: header plus the offset where tensor
// data begins.
type safetensorsFile struct {
	header map[string]json.RawMessage
	base   int
	raw    []byte
}

// isSafetensorsContainer reports whether data holds a safetensors container.
// JSON checkpoints begin with '{', so the two are never ambiguous.
func isSafetensorsContainer(data []byte) bool {
	if trimmed := bytes.TrimLeft(data, " \t\r\n"); len(trimmed) == 0 || trimmed[0] == '{' {
		return false
	}
	if len(data) < 9 {
		return false
	}
	headerLen := binary.LittleEndian.Uint64(data[:8])
	return headerLen > 0 && headerLen <= uint64(len(data)) && 8+int(headerLen) <= len(data) && data[8] == '{'
}

func parseSafetensors(data []byte) (*safetensorsFile, error) {
	if len(data) < 9 {
		return nil, fmt.Errorf("safetensors file too small")
	}
	headerLen := binary.LittleEndian.Uint64(data[:8])
	if headerLen == 0 || headerLen > uint64(len(data)) || 8+int(headerLen) > len(data) {
		return nil, fmt.Errorf("safetensors header length %d out of bounds", headerLen)
	}
	header := make(map[string]json.RawMessage)
	if err := json.Unmarshal(data[8:8+headerLen], &header); err != nil {
		return nil, fmt.Errorf("decode safetensors header: %w", err)
	}
	file := &safetensorsFile{header: header, base: 8 + int(headerLen), raw: data}
	if err := file.validateLayout(); err != nil {
		return nil, err
	}
	return file, nil
}

// validateLayout checks the tensor table as a whole: well-formed entries,
// byte ranges inside the data region, no overlapping tensors, and a data
// region covered exactly by the declared tensors, so truncation and trailing
// garbage are both caught before any tensor is read.
func (f *safetensorsFile) validateLayout() error {
	type span struct {
		name       string
		start, end int
	}
	region := len(f.raw) - f.base
	spans := make([]span, 0, len(f.header))
	maxEnd := 0
	for name, raw := range f.header {
		if name == "__metadata__" {
			continue
		}
		var entry stHeaderEntry
		if err := json.Unmarshal(raw, &entry); err != nil {
			return fmt.Errorf("safetensors tensor %q header: %w", name, err)
		}
		if len(entry.DataOffsets) != 2 {
			return fmt.Errorf("safetensors tensor %q data offsets %v", name, entry.DataOffsets)
		}
		start, end := entry.DataOffsets[0], entry.DataOffsets[1]
		if start < 0 || end < start {
			return fmt.Errorf("safetensors tensor %q data offsets %v", name, entry.DataOffsets)
		}
		if end > region {
			return fmt.Errorf("safetensors tensor %q ends at %d, past the %d-byte data region", name, end, region)
		}
		elems := 1
		for _, d := range entry.Shape {
			if d < 0 {
				return fmt.Errorf("safetensors tensor %q has negative dimension %v", name, entry.Shape)
			}
			if d > 0 && elems > int(^uint(0)>>1)/d {
				return fmt.Errorf("safetensors tensor %q shape %v overflows", name, entry.Shape)
			}
			elems *= d
		}
		if entry.Dtype == "F32" && end-start != elems*4 {
			return fmt.Errorf("safetensors tensor %q spans %d bytes, shape %v of F32 needs %d",
				name, end-start, entry.Shape, elems*4)
		}
		spans = append(spans, span{name: name, start: start, end: end})
		if end > maxEnd {
			maxEnd = end
		}
	}
	sort.Slice(spans, func(i, j int) bool { return spans[i].start < spans[j].start })
	for i := 1; i < len(spans); i++ {
		if spans[i].start < spans[i-1].end {
			return fmt.Errorf("safetensors tensors %q and %q overlap", spans[i-1].name, spans[i].name)
		}
	}
	if maxEnd != region {
		return fmt.Errorf("safetensors data region is %d bytes but the tensors cover %d (truncated or trailing data)",
			region, maxEnd)
	}
	return nil
}

func (f *safetensorsFile) metadata() map[string]string {
	meta := make(map[string]string)
	if raw, ok := f.header["__metadata__"]; ok {
		_ = json.Unmarshal(raw, &meta)
	}
	return meta
}

// copyInto validates one tensor and copies its values into target.
func (f *safetensorsFile) copyInto(name string, target []float32) error {
	entryRaw, ok := f.header[name]
	if !ok {
		return fmt.Errorf("safetensors missing tensor %q", name)
	}
	var entry stHeaderEntry
	if err := json.Unmarshal(entryRaw, &entry); err != nil {
		return fmt.Errorf("bad header for %q: %w", name, err)
	}
	if entry.Dtype != "F32" {
		return fmt.Errorf("tensor %q dtype %q, want F32", name, entry.Dtype)
	}
	if len(entry.Shape) != 1 || entry.Shape[0] != len(target) {
		return fmt.Errorf("tensor %q shape %v, want [%d]", name, entry.Shape, len(target))
	}
	if len(entry.DataOffsets) != 2 {
		return fmt.Errorf("tensor %q data offsets %v", name, entry.DataOffsets)
	}
	start, end := entry.DataOffsets[0], entry.DataOffsets[1]
	if start < 0 || end-start != len(target)*4 || f.base+end > len(f.raw) {
		return fmt.Errorf("tensor %q data offsets %v out of bounds", name, entry.DataOffsets)
	}
	payload := f.raw[f.base+start : f.base+end]
	for i := range target {
		target[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[i*4:]))
	}
	return nil
}

// LoadSafetensors reads a weights-only file and copies values into the model's
// parameters in parameterSpecs order.
func (m *Model) LoadSafetensors(path string) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	file, err := parseSafetensors(raw)
	if err != nil {
		return err
	}
	for _, s := range m.parameterSpecs() {
		if err := file.copyInto(s.Name, s.P.values); err != nil {
			return err
		}
	}
	if err := m.validateModelState(); err != nil {
		return fmt.Errorf("weights integrity: %w", err)
	}
	return nil
}

// loadSafetensorsCheckpoint rebuilds a model, tokenizer and training state from
// a SaveSafetensorsCheckpoint container.
func loadSafetensorsCheckpoint(data []byte) (*Model, error) {
	file, err := parseSafetensors(data)
	if err != nil {
		return nil, err
	}
	meta := file.metadata()
	if meta["format"] != safetensorsCheckpointFormat {
		return nil, fmt.Errorf("safetensors file is not a full checkpoint (format %q)", meta["format"])
	}
	if v := meta["format_version"]; v != "" && v != "1" {
		return nil, fmt.Errorf("checkpoint format version %q is not supported (this build reads version 1)", v)
	}
	switch meta["has_optimizer"] {
	case "", "0", "1":
	default:
		return nil, fmt.Errorf("checkpoint has_optimizer=%q, want \"0\" or \"1\"", meta["has_optimizer"])
	}
	var config Config
	if err := json.Unmarshal([]byte(meta["model_config"]), &config); err != nil {
		return nil, fmt.Errorf("checkpoint config: %w", err)
	}
	var parts checkpointTokenizer
	if err := json.Unmarshal([]byte(meta["tokenizer"]), &parts); err != nil {
		return nil, fmt.Errorf("checkpoint tokenizer: %w", err)
	}
	tokenizer := tokenizerFromCheckpoint(parts.Tokens, parts.Merges, parts.SpecialTokens)
	if hash := meta["tokenizer_hash"]; hash != "" && TokenizerFingerprint(tokenizer) != hash {
		return nil, fmt.Errorf("checkpoint tokenizer hash %q does not match serialized tokenizer", hash)
	}
	model, err := NewModel(config, tokenizer, 1)
	if err != nil {
		return nil, err
	}
	specs := model.parameterSpecs()
	if pc := meta["parameter_count"]; pc != "" {
		declared, err := strconv.Atoi(pc)
		if err != nil {
			return nil, fmt.Errorf("checkpoint parameter_count %q is not a number", pc)
		}
		if declared != len(specs) {
			return nil, fmt.Errorf("checkpoint parameter_count %d does not match the model's %d parameters", declared, len(specs))
		}
	}
	for _, s := range specs {
		if err := file.copyInto(s.Name, s.P.values); err != nil {
			return nil, err
		}
	}
	if meta["has_optimizer"] == "1" {
		for _, s := range specs {
			if err := file.copyInto("optimizer.first."+s.Name, s.P.first); err != nil {
				return nil, err
			}
			if err := file.copyInto("optimizer.second."+s.Name, s.P.second); err != nil {
				return nil, err
			}
		}
	}
	if raw := meta["step"]; raw != "" {
		step, err := strconv.ParseInt(raw, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("checkpoint step %q: %w", raw, err)
		}
		model.step = step
	}
	if raw := meta["seed"]; raw != "" {
		seed, err := strconv.ParseInt(raw, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("checkpoint seed %q: %w", raw, err)
		}
		model.seed = seed
	}
	if err := model.validateModelState(); err != nil {
		return nil, fmt.Errorf("checkpoint integrity: %w", err)
	}
	return model, nil
}
