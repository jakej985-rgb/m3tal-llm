// Package memplan estimates inference memory before a model or request
// allocates it.
//
// The planner is pure arithmetic: it turns a model shape plus a requested
// batch/context into an itemised Requirements (weights, quantization
// metadata, KV cache, scratch buffers and safety headroom) and compares that
// with a memory budget. The budget can come from a Provider that reports free
// device memory, from an explicit user cap, or from neither — in which case
// the resulting plan is advisory only and nothing is refused.
//
// Nothing here allocates memory or knows about CUDA. A caller (for example
// the inference backend) translates its configuration, calls Plan before
// uploading weights or creating a KV cache, and decides how to report or act
// on the Decision.
package memplan

import (
	"errors"
	"fmt"
	"math"
)

// DType is the storage dtype assumed for a memory component.
type DType uint8

const (
	F32 DType = iota
	F16
	INT8
	INT4
)

// BitsPerElement reports the storage width of the dtype, or 0 when unknown.
func (d DType) BitsPerElement() int {
	switch d {
	case F32:
		return 32
	case F16:
		return 16
	case INT8:
		return 8
	case INT4:
		return 4
	}
	return 0
}

func (d DType) String() string {
	switch d {
	case F32:
		return "f32"
	case F16:
		return "f16"
	case INT8:
		return "int8"
	case INT4:
		return "int4"
	}
	return fmt.Sprintf("dtype(%d)", uint8(d))
}

// Planner defaults. Zero values in Config select these.
const (
	// DefaultHeadroomFraction is reserved on top of the estimated working
	// set so allocator fragmentation and unreported runtime state do not
	// turn a "fits" estimate into an out-of-memory failure.
	DefaultHeadroomFraction = 0.10
	// DefaultActivationBuffers is the number of width-sized activation
	// buffers retained per layer (norm outputs, Q/K/V, attention output,
	// residuals, SwiGLU gate/up/activated, projections). It is calibrated
	// against measured CPU forward allocations in
	// internal/lm/scratch_calibration_test.go: ~20.2 buffers per layer were
	// observed for a 32-token, 32-wide model, and 28 leaves headroom for
	// allocator bookkeeping. The batch-forward path retains every layer's
	// buffers, so this term scales with Layers; KV generation is
	// conservatively over-reserved.
	DefaultActivationBuffers = 28
	// DefaultScaleBits is the per-group quant scale width (fp16).
	DefaultScaleBits = 16
)

// Errors reported by Estimate and Plan.
var (
	ErrInvalidConfig      = errors.New("memplan: invalid config")
	ErrOverflow           = errors.New("memplan: memory estimate overflows int64")
	ErrInsufficientMemory = errors.New("memplan: insufficient memory")
	// ErrUnsupported is returned by a Provider that cannot report free
	// memory (for example a CPU-only build). Plan treats it as an unknown
	// budget and produces an advisory plan.
	ErrUnsupported = errors.New("memplan: provider cannot report available bytes")
)

// Config describes the model shape and storage choices. It does not depend on
// any particular model package.
type Config struct {
	// Parameters is the number of stored inference parameters (all weights,
	// excluding KV cache and activations).
	Parameters int64
	Layers     int
	Width      int
	Heads      int
	Vocab      int

	WeightDType     DType
	KVDType         DType
	ActivationDType DType

	// QuantGroupSize is the number of weights sharing one quantization
	// scale. Zero means one per-tensor scale; negative values are invalid.
	// Ignored for float weight dtypes.
	QuantGroupSize int
	// ScaleBits is the storage width of each quant scale. Zero selects
	// DefaultScaleBits. Ignored for float weight dtypes.
	ScaleBits int
	// ZeroPointBits is per-group zero-point storage; zero means symmetric
	// quantization with no zero-point array.
	ZeroPointBits int

	// HeadroomFraction is reserved on top of the working set. Zero selects
	// DefaultHeadroomFraction; a negative value disables headroom.
	HeadroomFraction float64
	// ActivationBuffers is the number of width-sized activation buffers
	// retained per layer. Zero selects DefaultActivationBuffers.
	ActivationBuffers int
}

// Request is one inference configuration to plan for.
type Request struct {
	Config        Config
	Batch         int
	ContextTokens int
}

// Requirements is an itemised estimate in bytes. All fields are exact
// integers; no allocation has happened when it is produced.
type Requirements struct {
	WeightsBytes       int64
	QuantMetadataBytes int64
	KVCacheBytes       int64
	ScratchBytes       int64
	HeadroomBytes      int64
	TotalBytes         int64
}

func (r Requirements) String() string {
	return fmt.Sprintf(
		"weights=%dB metadata=%dB kv=%dB scratch=%dB headroom=%dB total=%dB",
		r.WeightsBytes, r.QuantMetadataBytes, r.KVCacheBytes,
		r.ScratchBytes, r.HeadroomBytes, r.TotalBytes)
}

// ActualAllocation describes memory measured after a backend has allocated.
// KVCacheBytes and ScratchBytes are zero when those buffers are created per
// request instead of at load time. HostBytes covers host-side copies the
// estimate does not model, such as a CUDA fallback clone.
type ActualAllocation struct {
	WeightsBytes int64
	KVCacheBytes int64
	ScratchBytes int64
	HostBytes    int64
}

// ConstructionBytes is the total measured at load time.
func (a ActualAllocation) ConstructionBytes() int64 {
	return a.WeightsBytes + a.KVCacheBytes + a.ScratchBytes + a.HostBytes
}

// Comparison relates an estimate to a measured allocation. WeightDeltaBytes
// is the diagnostic that matters most: a non-zero value means the model to
// planner mapping has drifted from what the backend really allocates.
type Comparison struct {
	Estimated         Requirements
	Actual            ActualAllocation
	WeightDeltaBytes  int64
	HostOverheadBytes int64
}

// Compare pairs an estimate with measured allocation without judging it; the
// caller decides whether a delta is acceptable. KV cache and scratch are
// per-request buffers in the current backends, so a construction-time
// comparison normally leaves them zero.
func Compare(estimated Requirements, actual ActualAllocation) Comparison {
	return Comparison{
		Estimated:         estimated,
		Actual:            actual,
		WeightDeltaBytes:  actual.WeightsBytes - estimated.WeightsBytes,
		HostOverheadBytes: actual.HostBytes,
	}
}

func (c Comparison) String() string {
	return fmt.Sprintf(
		"estimated_weights=%dB actual_weights=%dB weight_delta=%dB host_copy=%dB actual_at_load=%dB estimated_total=%dB",
		c.Estimated.WeightsBytes, c.Actual.WeightsBytes, c.WeightDeltaBytes,
		c.HostOverheadBytes, c.Actual.ConstructionBytes(), c.Estimated.TotalBytes)
}

// Provider reports free memory (device or otherwise) in bytes. Availability
// is queried per plan, never cached by this package.
type Provider interface {
	AvailableBytes() (int64, error)
}

// Limits bounds a plan.
type Limits struct {
	// Provider optionally reports free memory. Nil means no query.
	Provider Provider
	// UserCapBytes is an explicit upper bound; zero means no cap. When both
	// are present the smaller budget wins.
	UserCapBytes int64
	// MinContextTokens is the smallest context the planner may fall back
	// to. Zero means 1.
	MinContextTokens int
}

// Source records where the budget came from.
type Source string

const (
	SourceProvider Source = "provider"
	SourceUserCap  Source = "user_cap"
	SourceUnknown  Source = "unknown"
)

// Decision is the outcome of Plan. Admitted is the configuration that fits
// the budget (equal to Requested when nothing had to change, and also when
// the budget is unknown and the plan is advisory).
type Decision struct {
	Requested             Request
	Admitted              Request
	RequestedRequirements Requirements
	Requirements          Requirements
	// AvailableBytes is the budget used, or -1 when no source reported one.
	AvailableBytes int64
	Source         Source
	ReducedBatch   bool
	ReducedContext bool
	// Advisory is set when no budget was available: the plan is an estimate
	// only and nothing was refused.
	Advisory bool
	// ProviderError records a failed availability query, if any.
	ProviderError error
	Reason        string
}

// Estimate returns the itemised memory requirement for req without querying
// any provider or allocating anything.
//
// Formulas:
//   - weights     = parameters * bits(WeightDType), rounded up to a byte
//   - metadata    = ceil(parameters / group) * (ScaleBits+ZeroPointBits)
//     only for INT8/INT4 weights
//   - KV cache    = 2 (K and V) * layers * batch * context * width
//   - bits(KVDType)
//   - scratch     = batch*context*width*ActivationBuffers*layers
//   - 2*batch*context*vocab (logits and probabilities)
//   - batch*heads*context*context (attention scores)
//     all at bits(ActivationDType); the per-layer term is calibrated
//     against measured allocations rather than assumed
//   - headroom    = ceil((weights+metadata+kv+scratch) * HeadroomFraction)
func Estimate(req Request) (Requirements, error) {
	cfg, err := req.Config.normalize()
	if err != nil {
		return Requirements{}, err
	}
	if req.Batch <= 0 {
		return Requirements{}, fmt.Errorf("%w: batch must be positive, got %d", ErrInvalidConfig, req.Batch)
	}
	if req.ContextTokens <= 0 {
		return Requirements{}, fmt.Errorf("%w: context tokens must be positive, got %d", ErrInvalidConfig, req.ContextTokens)
	}

	weights, err := bytesFor(cfg.Parameters, cfg.WeightDType.BitsPerElement())
	if err != nil {
		return Requirements{}, fmt.Errorf("weights: %w", err)
	}

	var metadata int64
	if cfg.WeightDType == INT8 || cfg.WeightDType == INT4 {
		group := int64(cfg.QuantGroupSize)
		if group <= 0 {
			group = cfg.Parameters
			if group == 0 {
				group = 1
			}
		}
		groups := cfg.Parameters / group
		if cfg.Parameters%group != 0 {
			groups++
		}
		metadata, err = bytesFor(groups, cfg.ScaleBits+cfg.ZeroPointBits)
		if err != nil {
			return Requirements{}, fmt.Errorf("quantization metadata: %w", err)
		}
	}

	kvElements, err := mulChain(2, int64(cfg.Layers), int64(req.Batch), int64(req.ContextTokens), int64(cfg.Width))
	if err != nil {
		return Requirements{}, fmt.Errorf("kv cache: %w", err)
	}
	kvBytes, err := bytesFor(kvElements, cfg.KVDType.BitsPerElement())
	if err != nil {
		return Requirements{}, fmt.Errorf("kv cache: %w", err)
	}

	activationElements, err := mulChain(int64(req.Batch), int64(req.ContextTokens), int64(cfg.Width), int64(cfg.ActivationBuffers), int64(cfg.Layers))
	if err != nil {
		return Requirements{}, fmt.Errorf("scratch activations: %w", err)
	}
	logitElements, err := mulChain(2, int64(req.Batch), int64(req.ContextTokens), int64(cfg.Vocab))
	if err != nil {
		return Requirements{}, fmt.Errorf("scratch logits: %w", err)
	}
	attentionElements, err := mulChain(int64(req.Batch), int64(cfg.Heads), int64(req.ContextTokens), int64(req.ContextTokens))
	if err != nil {
		return Requirements{}, fmt.Errorf("scratch attention: %w", err)
	}
	scratchElements, err := add(activationElements, logitElements, attentionElements)
	if err != nil {
		return Requirements{}, fmt.Errorf("scratch: %w", err)
	}
	scratchBytes, err := bytesFor(scratchElements, cfg.ActivationDType.BitsPerElement())
	if err != nil {
		return Requirements{}, fmt.Errorf("scratch: %w", err)
	}

	working, err := add(weights, metadata, kvBytes, scratchBytes)
	if err != nil {
		return Requirements{}, err
	}
	var headroom int64
	if cfg.HeadroomFraction > 0 {
		reserved := math.Ceil(float64(working) * cfg.HeadroomFraction)
		if reserved > float64(math.MaxInt64) {
			return Requirements{}, fmt.Errorf("%w: headroom", ErrOverflow)
		}
		headroom = int64(reserved)
	}
	total, err := add(working, headroom)
	if err != nil {
		return Requirements{}, err
	}

	return Requirements{
		WeightsBytes:       weights,
		QuantMetadataBytes: metadata,
		KVCacheBytes:       kvBytes,
		ScratchBytes:       scratchBytes,
		HeadroomBytes:      headroom,
		TotalBytes:         total,
	}, nil
}

// Plan estimates req, resolves a budget, and returns the largest request that
// fits. When the requested batch/context does not fit it reduces the batch to
// 1 first, then shrinks the context (binary search) down to
// Limits.MinContextTokens. If even that does not fit, Plan returns
// ErrInsufficientMemory together with a Decision describing the smallest
// configuration it tried.
//
// When no budget is available (nil Provider, provider error, or no user cap)
// the plan is advisory: Admitted equals Requested and nothing is refused.
func Plan(req Request, limits Limits) (Decision, error) {
	requested, err := Estimate(req)
	if err != nil {
		return Decision{}, err
	}

	minContext := limits.MinContextTokens
	if minContext <= 0 {
		minContext = 1
	}
	if minContext > req.ContextTokens {
		minContext = req.ContextTokens
	}

	budget, source, providerErr := resolveBudget(limits)
	decision := Decision{
		Requested:             req,
		RequestedRequirements: requested,
		AvailableBytes:        -1,
		Source:                source,
		ProviderError:         providerErr,
	}

	if source == SourceUnknown {
		decision.Admitted = req
		decision.Requirements = requested
		decision.Advisory = true
		decision.Reason = "available memory unknown; plan is an estimate only"
		if providerErr != nil {
			decision.Reason = fmt.Sprintf("%s (provider: %v)", decision.Reason, providerErr)
		}
		return decision, nil
	}
	decision.AvailableBytes = budget

	admit := func(admitted Request, reducedBatch, reducedContext bool, reason string) (Decision, error) {
		requirements, err := Estimate(admitted)
		if err != nil {
			return Decision{}, err
		}
		decision.Admitted = admitted
		decision.Requirements = requirements
		decision.ReducedBatch = reducedBatch
		decision.ReducedContext = reducedContext
		decision.Reason = reason
		return decision, nil
	}

	if requested.TotalBytes <= budget {
		return admit(req, false, false, "fits as requested")
	}

	// A smaller batch is the least surprising reduction: context and output
	// length stay identical.
	if req.Batch > 1 {
		single := req
		single.Batch = 1
		if requirements, err := Estimate(single); err != nil {
			return Decision{}, err
		} else if requirements.TotalBytes <= budget {
			return admit(single, true, false, "batch reduced to 1 to fit")
		}
	}

	// Shrink the context at batch 1. Requirements are monotonic in context,
	// so binary search for the largest context that fits.
	base := req
	base.Batch = 1
	low, high := minContext, req.ContextTokens
	best := -1
	for low <= high {
		mid := low + (high-low)/2
		candidate := base
		candidate.ContextTokens = mid
		requirements, err := Estimate(candidate)
		if err != nil {
			return Decision{}, err
		}
		if requirements.TotalBytes <= budget {
			best = mid
			low = mid + 1
		} else {
			high = mid - 1
		}
	}
	if best > 0 {
		candidate := base
		candidate.ContextTokens = best
		return admit(candidate, req.Batch > 1, best != req.ContextTokens,
			fmt.Sprintf("context reduced from %d to %d to fit", req.ContextTokens, best))
	}

	smallest := base
	smallest.ContextTokens = minContext
	minimum, err := Estimate(smallest)
	if err != nil {
		return Decision{}, err
	}
	decision.Admitted = smallest
	decision.Requirements = minimum
	decision.ReducedBatch = req.Batch > 1
	decision.ReducedContext = minContext != req.ContextTokens
	decision.Reason = fmt.Sprintf(
		"minimum configuration (batch 1, context %d) needs %d bytes, only %d available",
		minContext, minimum.TotalBytes, budget)
	return decision, fmt.Errorf("%w: %s", ErrInsufficientMemory, decision.Reason)
}

// resolveBudget picks the strictest available limit. Provider failures are
// reported but do not abort planning: an explicit cap can still be used, and
// otherwise the plan becomes advisory.
func resolveBudget(limits Limits) (int64, Source, error) {
	budget, source := int64(0), SourceUnknown
	if limits.UserCapBytes > 0 {
		budget, source = limits.UserCapBytes, SourceUserCap
	}
	if limits.Provider == nil {
		return budget, source, nil
	}
	available, err := limits.Provider.AvailableBytes()
	if err != nil {
		return budget, source, err
	}
	if available < 0 {
		return budget, source, fmt.Errorf("%w: negative availability %d", ErrInvalidConfig, available)
	}
	if source == SourceUnknown || available < budget {
		return available, SourceProvider, nil
	}
	return budget, source, nil
}

func (c Config) normalize() (Config, error) {
	if c.Parameters < 0 {
		return c, fmt.Errorf("%w: parameters must be non-negative", ErrInvalidConfig)
	}
	if c.Layers <= 0 || c.Width <= 0 || c.Heads <= 0 || c.Vocab <= 0 {
		return c, fmt.Errorf("%w: layers, width, heads and vocab must be positive", ErrInvalidConfig)
	}
	if c.WeightDType.BitsPerElement() == 0 {
		return c, fmt.Errorf("%w: unknown weight dtype %s", ErrInvalidConfig, c.WeightDType)
	}
	if c.KVDType.BitsPerElement() == 0 {
		return c, fmt.Errorf("%w: unknown kv dtype %s", ErrInvalidConfig, c.KVDType)
	}
	if c.ActivationDType.BitsPerElement() == 0 {
		return c, fmt.Errorf("%w: unknown activation dtype %s", ErrInvalidConfig, c.ActivationDType)
	}
	if c.QuantGroupSize < 0 {
		return c, fmt.Errorf("%w: quant group size must not be negative", ErrInvalidConfig)
	}
	if c.ZeroPointBits < 0 {
		return c, fmt.Errorf("%w: zero-point bits must not be negative", ErrInvalidConfig)
	}
	if c.ScaleBits < 0 {
		return c, fmt.Errorf("%w: scale bits must not be negative", ErrInvalidConfig)
	}
	if c.ActivationBuffers < 0 {
		return c, fmt.Errorf("%w: activation buffers must not be negative", ErrInvalidConfig)
	}
	if c.HeadroomFraction > 1 {
		return c, fmt.Errorf("%w: headroom fraction must be <= 1", ErrInvalidConfig)
	}

	if c.HeadroomFraction == 0 {
		c.HeadroomFraction = DefaultHeadroomFraction
	}
	if c.ActivationBuffers == 0 {
		c.ActivationBuffers = DefaultActivationBuffers
	}
	if c.ScaleBits == 0 {
		c.ScaleBits = DefaultScaleBits
	}
	return c, nil
}

func mulChain(factors ...int64) (int64, error) {
	product := int64(1)
	for _, factor := range factors {
		var err error
		product, err = mul(product, factor)
		if err != nil {
			return 0, err
		}
	}
	return product, nil
}

func mul(a, b int64) (int64, error) {
	if a < 0 || b < 0 {
		return 0, fmt.Errorf("%w: negative factor %d*%d", ErrOverflow, a, b)
	}
	if a != 0 && b > math.MaxInt64/a {
		return 0, fmt.Errorf("%w: %d*%d", ErrOverflow, a, b)
	}
	return a * b, nil
}

func add(parts ...int64) (int64, error) {
	total := int64(0)
	for _, part := range parts {
		if part < 0 {
			return 0, fmt.Errorf("%w: negative addend %d", ErrOverflow, part)
		}
		if total > math.MaxInt64-part {
			return 0, fmt.Errorf("%w: sum", ErrOverflow)
		}
		total += part
	}
	return total, nil
}

func bytesFor(count int64, bits int) (int64, error) {
	if bits <= 0 {
		return 0, fmt.Errorf("%w: unknown dtype width", ErrInvalidConfig)
	}
	totalBits, err := mul(count, int64(bits))
	if err != nil {
		return 0, err
	}
	if totalBits > math.MaxInt64-7 {
		return 0, fmt.Errorf("%w: byte rounding", ErrOverflow)
	}
	return (totalBits + 7) / 8, nil
}
