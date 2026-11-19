package lm

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"unsafe"
)

// Config defines the multi-layer, multi-head decoder-only Transformer.
type Config struct {
	Context int `json:"context"`
	Width   int `json:"width"`
	Layers  int `json:"layers,omitempty"`
	Heads   int `json:"heads,omitempty"`
}

// Normalize ensures Layers and Heads have valid defaults.
func (c *Config) Normalize() {
	if c.Layers <= 0 {
		c.Layers = 1
	}
	if c.Heads <= 0 {
		c.Heads = 1
	}
}

// ParameterCount returns the analytic trainable-parameter count for this
// config and vocabulary size. It mirrors NewModel's layout exactly:
// token embedding (vocab*width), per-block RMSNorm gammas (2*width),
// attention Q/K/V/O projections (4*(width^2+width)), SwiGLU FFN
// (gate/up width->2*width, down 2*width->width = 6*width^2+5*width),
// and the output projection (width*vocab+vocab). Position encoding is
// RoPE, so there is no position table.
func (c Config) ParameterCount(vocab int) int {
	c.Normalize()
	w := c.Width
	perBlock := 10*w*w + 11*w
	return vocab*w + c.Layers*perBlock + w*vocab + vocab
}

// ParameterBreakdown splits ParameterCount into embedding, per-block,
// and output-projection components for reporting.
func (c Config) ParameterBreakdown(vocab int) (emb, blocks, output int) {
	c.Normalize()
	w := c.Width
	emb = vocab * w
	blocks = c.Layers * (10*w*w + 11*w)
	output = w*vocab + vocab
	return emb, blocks, output
}

// DefaultConfig fits an instruction-capable training prototype on CPU-only machines.
func DefaultConfig() Config { return Config{Context: 256, Width: 64, Layers: 4, Heads: 4} }

// parameter owns contiguous float32 values, gradients, and Adam state.
type parameter struct {
	values   []float32
	gradient []float32
	first    []float32
	second   []float32
}

type linear struct {
	weights *parameter // row-major [input, output]
	bias    *parameter
	input   int
	output  int
}

// transformerBlock holds one decoder layer's parameters.
// normAttn and normFF are the learnable RMSNorm gamma vectors applied
// before the attention sub-layer and before the feed-forward sub-layer
// respectively (pre-norm architecture).
type transformerBlock struct {
	normAttn *parameter // RMSNorm gamma, pre-attention [width]
	normFF   *parameter // RMSNorm gamma, pre-FFN       [width]
	query    linear
	key      linear
	value    linear
	attnOut  linear
	ffGate   linear // SwiGLU gate projection [width, width*2]
	ffUp     linear // SwiGLU up projection   [width, width*2]
	ffDown   linear // SwiGLU down projection [width*2, width]
}

// Model is a causal language model trained and run with contiguous float32
// parameters. Position is encoded via RoPE inside each attention block;
// no learned position embedding table is maintained.
type Model struct {
	Config     Config
	Tokenizer  *Tokenizer
	tokenEmb   *parameter // [vocabulary, width]
	blocks     []transformerBlock
	output     linear
	parameters []*parameter
	step       int64
	seed       int64

	// backend is the cached inference snapshot. It is built lazily and may
	// be requested by several HTTP handlers at once, so access is guarded
	// by backendMu. Training invalidates it via invalidateBackend.
	backendMu sync.Mutex
	backend   inferenceBackend
}

// ParameterCount reports the number of trainable scalar weights.
func (m *Model) ParameterCount() int {
	count := 0
	for _, p := range m.parameters {
		count += len(p.values)
	}
	return count
}

func newParameter(size int) *parameter {
	return &parameter{
		values:   make([]float32, size),
		gradient: make([]float32, size),
		first:    make([]float32, size),
		second:   make([]float32, size),
	}
}

func (m *Model) addParameter(p *parameter) *parameter {
	m.parameters = append(m.parameters, p)
	return p
}

// NewModel initializes all weights with a deterministic seed.
// Every transformer block now carries two RMSNorm gamma vectors (pre-attention
// and pre-FFN). Position is encoded via RoPE; no posEmb table is allocated.
func NewModel(config Config, tokenizer *Tokenizer, seed int64) (*Model, error) {
	// Validate before defaulting so a negative layers/heads value in a
	// checkpoint is rejected instead of silently becoming the default.
	if err := config.Validate(); err != nil {
		return nil, err
	}
	config.Normalize()
	if err := tokenizer.Validate(); err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(seed))
	m := &Model{Config: config, Tokenizer: tokenizer, seed: seed}
	newMatrix := func(rows, cols int) *parameter {
		p := newParameter(rows * cols)
		limit := float32(1 / math.Sqrt(float64(cols)))
		for i := range p.values {
			p.values[i] = (rng.Float32()*2 - 1) * limit
		}
		return m.addParameter(p)
	}
	newLinear := func(input, output int) linear {
		weights := newMatrix(input, output)
		bias := m.addParameter(newParameter(output))
		return linear{weights: weights, bias: bias, input: input, output: output}
	}
	newNormGamma := func(width int) *parameter {
		p := m.addParameter(newParameter(width))
		for i := range p.values {
			p.values[i] = 1.0 // identity: no scaling at the start of training
		}
		return p
	}

	width, vocab := config.Width, len(tokenizer.Tokens)
	m.tokenEmb = newMatrix(vocab, width)
	// No posEmb — RoPE encodes position inside each block's attention.

	m.blocks = make([]transformerBlock, config.Layers)
	for l := 0; l < config.Layers; l++ {
		m.blocks[l] = transformerBlock{
			normAttn: newNormGamma(width),
			normFF:   newNormGamma(width),
			query:    newLinear(width, width),
			key:      newLinear(width, width),
			value:    newLinear(width, width),
			attnOut:  newLinear(width, width),
			ffGate:   newLinear(width, width*2),
			ffUp:     newLinear(width, width*2),
			ffDown:   newLinear(width*2, width),
		}
	}
	m.output = newLinear(width, vocab)
	return m, nil
}

func (m *Model) parameterView() inferenceWeights {
	m.Config.Normalize()
	blocks := make([]inferenceBlock, len(m.blocks))
	for i, b := range m.blocks {
		blocks[i] = inferenceBlock{
			normAttn: b.normAttn.values,
			normFF:   b.normFF.values,
			query:    inferenceLayer{weights: b.query.weights.values, bias: b.query.bias.values},
			key:      inferenceLayer{weights: b.key.weights.values, bias: b.key.bias.values},
			value:    inferenceLayer{weights: b.value.weights.values, bias: b.value.bias.values},
			attnOut:  inferenceLayer{weights: b.attnOut.weights.values, bias: b.attnOut.bias.values},
			ffGate:   inferenceLayer{weights: b.ffGate.weights.values, bias: b.ffGate.bias.values},
			ffUp:     inferenceLayer{weights: b.ffUp.weights.values, bias: b.ffUp.bias.values},
			ffDown:   inferenceLayer{weights: b.ffDown.weights.values, bias: b.ffDown.bias.values},
		}
	}
	return inferenceWeights{
		tokenEmb: m.tokenEmb.values,
		blocks:   blocks,
		output:   inferenceLayer{weights: m.output.weights.values, bias: m.output.bias.values},
	}
}

func (m *Model) forward(ids []int) forwardCache {
	return forwardFloat32(m.Config, len(m.Tokenizer.Tokens), m.parameterView(), ids)
}

// bwdLinear32 computes input, weight, and bias gradients for one linear
// layer during backprop. x is [rows,inSize] activations, g is [rows,outSize]
// output gradients (already loss-scaled), w is row-major [inSize,outSize]
// forward weights. Weight/bias grads are accumulated (+=) into wGrad/bGrad;
// the input gradient is returned as a fresh [rows,inSize] slice.
//
// Parallelization preserves the serial accumulation order per element:
//   - dx rows are partitioned across workers (each (i,k) summed over j
//     in ascending order, same as serial).
//   - wGrad is partitioned by input dim k (each (k,j) summed over i in
//     ascending order, same as serial).
//   - bias grads stay serial (cheap, keeps order exact).
func bwdLinear32(x, g []float32, rows, inSize, outSize int, w, wGrad, bGrad []float32) []float32 {
	dx := make([]float32, rows*inSize)
	work := rows * inSize * outSize
	workers := runtime.GOMAXPROCS(0)
	if workers > 8 {
		workers = 8
	}
	if work < linearWorkThreshold || workers < 2 || inSize < workers || outSize < workers {
		bwdLinear32Serial(x, g, rows, inSize, outSize, w, wGrad, bGrad, dx)
		return dx
	}

	// Pass A: input gradient, partitioned by rows.
	rowChunk := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for wk := 0; wk < workers; wk++ {
		r0, r1 := wk*rowChunk, (wk+1)*rowChunk
		if r1 > rows {
			r1 = rows
		}
		if r0 >= r1 {
			break
		}
		wg.Add(1)
		go func(r0, r1 int) {
			defer wg.Done()
			for i := r0; i < r1; i++ {
				gRow := g[i*outSize : (i+1)*outSize]
				dxRow := dx[i*inSize : (i+1)*inSize]
				for k := 0; k < inSize; k++ {
					wRow := w[k*outSize : (k+1)*outSize]
					var acc float32
					for j := 0; j < outSize; j++ {
						acc += wRow[j] * gRow[j]
					}
					dxRow[k] = acc
				}
			}
		}(r0, r1)
	}
	wg.Wait()

	// Pass B: weight gradient, partitioned by input dim k. Each worker
	// owns whole rows of wGrad so writes are contiguous and race-free.
	kChunk := (inSize + workers - 1) / workers
	for wk := 0; wk < workers; wk++ {
		k0, k1 := wk*kChunk, (wk+1)*kChunk
		if k1 > inSize {
			k1 = inSize
		}
		if k0 >= k1 {
			break
		}
		wg.Add(1)
		go func(k0, k1 int) {
			defer wg.Done()
			for k := k0; k < k1; k++ {
				wgRow := wGrad[k*outSize : (k+1)*outSize]
				for i := 0; i < rows; i++ {
					xk := x[i*inSize+k]
					gRow := g[i*outSize : (i+1)*outSize]
					for j := 0; j < outSize; j++ {
						wgRow[j] += xk * gRow[j]
					}
				}
			}
		}(k0, k1)
	}
	wg.Wait()

	// Bias gradient: column sums of g (serial, cheap).
	for i := 0; i < rows; i++ {
		gRow := g[i*outSize : (i+1)*outSize]
		for j := 0; j < outSize; j++ {
			bGrad[j] += gRow[j]
		}
	}
	return dx
}

// bwdLinear32Serial is the reference single-threaded implementation.
func bwdLinear32Serial(x, g []float32, rows, inSize, outSize int, w, wGrad, bGrad, dx []float32) {
	for i := 0; i < rows; i++ {
		xRow := x[i*inSize : (i+1)*inSize]
		gRow := g[i*outSize : (i+1)*outSize]
		for k := 0; k < inSize; k++ {
			xk := xRow[k]
			wRow := w[k*outSize : (k+1)*outSize]
			wgRow := wGrad[k*outSize : (k+1)*outSize]
			var acc float32
			for j := 0; j < outSize; j++ {
				wgRow[j] += xk * gRow[j]
				acc += wRow[j] * gRow[j]
			}
			dx[i*inSize+k] = acc
		}
		for j := 0; j < outSize; j++ {
			bGrad[j] += gRow[j]
		}
	}
}

// backward propagates the cross-entropy loss gradient through the full model.
// If mask is non-nil, only positions where mask[i]==true contribute to the loss.
func (m *Model) backward(cache forwardCache, targets []int, mask []bool) {
	m.Config.Normalize()
	width, vocab, rows := m.Config.Width, len(m.Tokenizer.Tokens), len(cache.ids)
	heads := m.Config.Heads
	headDim := width / heads
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	activeCount := rows
	if mask != nil {
		activeCount = 0
		for i := 0; i < rows; i++ {
			if i < len(mask) && mask[i] {
				activeCount++
			}
		}
		if activeCount == 0 {
			return
		}
	}
	invRows := float32(1.0 / float64(activeCount))

	// ── 1. Logits → softmax gradient ──────────────────────────────────────
	gradLogits := make([]float32, rows*vocab)
	for i := 0; i < rows; i++ {
		if mask != nil && (i >= len(mask) || !mask[i]) {
			continue
		}
		for j := 0; j < vocab; j++ {
			gradLogits[i*vocab+j] = cache.probabilities[i*vocab+j] * invRows
		}
		gradLogits[i*vocab+targets[i]] -= invRows
	}

	// ── 2. Output linear layer ─────────────────────────────────────────────
	lastBlockIdx := len(m.blocks) - 1
	gradCurrentX := bwdLinear32(
		cache.blocks[lastBlockIdx].encoded, gradLogits,
		rows, width, vocab,
		m.output.weights.values, m.output.weights.gradient, m.output.bias.gradient,
	)

	// ── 3. Transformer blocks (reverse order) ─────────────────────────────
	for l := len(m.blocks) - 1; l >= 0; l-- {
		block := m.blocks[l]
		bc := cache.blocks[l]
		gradEncoded := gradCurrentX

		// ── 3a. FF residual split: encoded = residual + ff ────────────
		// gradResidual accumulates from the residual path; we start it with
		// gradEncoded (direct flow through the residual connection).
		gradResidual := make([]float32, rows*width)
		copy(gradResidual, gradEncoded)

		// ── 3b. SwiGLU: ffDown linear backward ────────────────────────
		// Forward: feedForward = activated · W_down + bias_down
		// Backward: dActivated[h] = Σ_d W_down[h*width+d] * dEncoded[d]
		hiddenDim := width * 2
		gradActivated := bwdLinear32(
			bc.ffActivated, gradEncoded,
			rows, hiddenDim, width,
			block.ffDown.weights.values, block.ffDown.weights.gradient, block.ffDown.bias.gradient,
		)

		// ── 3c. SwiGLU activation backward: activated = swish(gate) ⊙ up ──
		// dUp[h]   = dActivated[h] * swish(gate[h])
		// dGate[h] = dActivated[h] * up[h] * swishPrime(gate[h])
		gradGate := make([]float32, rows*hiddenDim)
		gradUp := make([]float32, rows*hiddenDim)
		for idx := 0; idx < rows*hiddenDim; idx++ {
			gradUp[idx] = gradActivated[idx] * swish(bc.ffGateOut[idx])
			gradGate[idx] = gradActivated[idx] * bc.ffUpOut[idx] * swishPrime(bc.ffGateOut[idx])
		}

		// ── 3d. ffGate and ffUp linear backward (input was normX2) ────
		// dNormX2[d] += Σ_h W_gate[d*hiddenDim+h]*dGate[h] + W_up[d*hiddenDim+h]*dUp[h]
		dxGate := bwdLinear32(bc.normX2, gradGate, rows, width, hiddenDim,
			block.ffGate.weights.values, block.ffGate.weights.gradient, block.ffGate.bias.gradient)
		dxUp := bwdLinear32(bc.normX2, gradUp, rows, width, hiddenDim,
			block.ffUp.weights.values, block.ffUp.weights.gradient, block.ffUp.bias.gradient)
		gradNormX2 := make([]float32, rows*width)
		for i := range gradNormX2 {
			gradNormX2[i] = dxGate[i] + dxUp[i]
		}

		// ── 3e. RMSNorm #2 backward (pre-FFN norm) ────────────────────
		// Input to norm was bc.residual; output was bc.normX2.
		gradRms2 := rmsNormBackward(gradNormX2, bc.residual, bc.normRms2, block.normFF.values, rows, width, block.normFF.gradient)
		for i := 0; i < rows*width; i++ {
			gradResidual[i] += gradRms2[i]
		}

		// ── 3f. Attn residual split: residual = x + attnProjection ────
		gradX := make([]float32, rows*width)
		copy(gradX, gradResidual) // direct flow through residual connection

		// ── 3g. attnOut linear ────────────────────────────────────────
		gradAttended := bwdLinear32(bc.attended, gradResidual, rows, width, width,
			block.attnOut.weights.values, block.attnOut.weights.gradient, block.attnOut.bias.gradient)

		// ── 3h. Multi-head causal attention backward ──────────────────
		// Gradients w.r.t. Q/K after RoPE, and V (no RoPE on V).
		gradQueryRope := make([]float32, rows*width)
		gradKeysRope := make([]float32, rows*width)
		gradValues := make([]float32, rows*width)

		gradAttention := make([]float32, heads*rows*rows)
		parallelFor(heads, rows*rows*headDim, func(h0, h1 int) {
			for h := h0; h < h1; h++ {
				headOffset := h * headDim
				headMatrixOffset := h * rows * rows
				for i := 0; i < rows; i++ {
					for d := 0; d < headDim; d++ {
						for j := 0; j <= i; j++ {
							gradAttention[headMatrixOffset+i*rows+j] += gradAttended[i*width+headOffset+d] * bc.v[j*width+headOffset+d]
							gradValues[j*width+headOffset+d] += bc.attention[headMatrixOffset+i*rows+j] * gradAttended[i*width+headOffset+d]
						}
					}
				}
			}
		})

		parallelFor(heads, rows*rows*headDim, func(h0, h1 int) {
			for h := h0; h < h1; h++ {
				headOffset := h * headDim
				headMatrixOffset := h * rows * rows
				for i := 0; i < rows; i++ {
					var weightedGrad float32
					for j := 0; j <= i; j++ {
						weightedGrad += bc.attention[headMatrixOffset+i*rows+j] * gradAttention[headMatrixOffset+i*rows+j]
					}
					for j := 0; j <= i; j++ {
						gradScore := bc.attention[headMatrixOffset+i*rows+j] * (gradAttention[headMatrixOffset+i*rows+j] - weightedGrad)
						for d := 0; d < headDim; d++ {
							// bc.q and bc.k are the post-RoPE vectors used in the forward pass.
							gradQueryRope[i*width+headOffset+d] += gradScore * bc.k[j*width+headOffset+d] * scale
							gradKeysRope[j*width+headOffset+d] += gradScore * bc.q[i*width+headOffset+d] * scale
						}
					}
				}
			}
		})

		// ── 3i. RoPE backward: inverse rotation on Q and K gradients ──
		applyInverseRoPE(gradQueryRope, gradKeysRope, rows, width, headDim, heads)

		// ── 3j. QKV linear projections backward (input was normX1) ────
		dxQ := bwdLinear32(bc.normX1, gradQueryRope, rows, width, width,
			block.query.weights.values, block.query.weights.gradient, block.query.bias.gradient)
		dxK := bwdLinear32(bc.normX1, gradKeysRope, rows, width, width,
			block.key.weights.values, block.key.weights.gradient, block.key.bias.gradient)
		dxV := bwdLinear32(bc.normX1, gradValues, rows, width, width,
			block.value.weights.values, block.value.weights.gradient, block.value.bias.gradient)
		gradNormX1 := make([]float32, rows*width)
		for i := range gradNormX1 {
			gradNormX1[i] = dxQ[i] + dxK[i] + dxV[i]
		}

		// ── 3k. RMSNorm #1 backward (pre-attention norm) ──────────────
		// Input to norm was bc.x; output was bc.normX1.
		gradRms1 := rmsNormBackward(gradNormX1, bc.x, bc.normRms1, block.normAttn.values, rows, width, block.normAttn.gradient)
		for i := 0; i < rows*width; i++ {
			gradX[i] += gradRms1[i]
		}

		gradCurrentX = gradX
	}

	// ── 4. Token embedding gradients (no posEmb) ──────────────────────────
	for row, id := range cache.ids {
		for d := 0; d < width; d++ {
			m.tokenEmb.gradient[id*width+d] += gradCurrentX[row*width+d]
		}
	}
}

func zeroGradients(parameters []*parameter) {
	for _, p := range parameters {
		clear(p.gradient)
	}
}

func (m *Model) adamStep(learningRate float32) { m.adamStepLR(learningRate, 1.0) }

// adamStepLR applies one Adam update with the given learning rate and
// global gradient-norm clip threshold. It returns the pre-clip gradient
// norm for logging.
func (m *Model) adamStepLR(learningRate float32, clipNorm float64) float64 {
	m.step++
	beta1, beta2 := float32(0.9), float32(0.999)
	bias1 := float32(1 - math.Pow(float64(beta1), float64(m.step)))
	bias2 := float32(1 - math.Pow(float64(beta2), float64(m.step)))

	var totalNormSq float64
	for _, p := range m.parameters {
		for _, g := range p.gradient {
			totalNormSq += float64(g * g)
		}
	}
	gradNorm := math.Sqrt(totalNormSq)
	clipScale := float32(1.0)
	if gradNorm > clipNorm {
		clipScale = float32(clipNorm / gradNorm)
	}

	for _, p := range m.parameters {
		for i, g := range p.gradient {
			gradient := g * clipScale
			p.first[i] = beta1*p.first[i] + (1-beta1)*gradient
			p.second[i] = beta2*p.second[i] + (1-beta2)*gradient*gradient
			first := p.first[i] / bias1
			second := p.second[i] / bias2
			p.values[i] -= learningRate * first / (float32(math.Sqrt(float64(second))) + 1e-8)
		}
	}
	return gradNorm
}

func tokenLoss32(logits []float32, target int) float64 {
	maximum := float32(math.Inf(-1))
	for _, value := range logits {
		if value > maximum {
			maximum = value
		}
	}
	var total float64
	for _, value := range logits {
		total += math.Exp(float64(value - maximum))
	}
	return math.Log(total) + float64(maximum) - float64(logits[target])
}

// meanWindowLoss returns average cross-entropy over active positions (all if mask is nil).
func meanWindowLoss(cache forwardCache, targets []int, mask []bool) float64 {
	if len(targets) == 0 || len(cache.ids) == 0 {
		return 0
	}
	rows := len(cache.ids)
	if len(targets) < rows {
		rows = len(targets)
	}
	vocab := len(cache.logits) / len(cache.ids)
	var totalLoss float64
	activeCount := 0
	for i := 0; i < rows; i++ {
		if mask != nil && (i >= len(mask) || !mask[i]) {
			continue
		}
		totalLoss += tokenLoss32(cache.logits[i*vocab:(i+1)*vocab], targets[i])
		activeCount++
	}
	if activeCount == 0 {
		return 0
	}
	return totalLoss / float64(activeCount)
}

// Train performs causal full-window training with Adam updates.
// Returns average pre-update cross-entropy per epoch.
func (m *Model) Train(texts []string, epochs, stepsPerEpoch int, learningRate float64) ([]float64, error) {
	if epochs <= 0 || stepsPerEpoch <= 0 || learningRate <= 0 || math.IsNaN(learningRate) || math.IsInf(learningRate, 0) {
		return nil, fmt.Errorf("epochs, steps, and finite positive learning rate are required")
	}
	m.invalidateBackend()
	var ids []int
	for _, text := range texts {
		ids = append(ids, m.Tokenizer.Encode(text)...)
		ids = append(ids, m.Tokenizer.Encode("\n")...)
	}
	if len(ids) < 3 {
		return nil, fmt.Errorf("training corpus must encode to at least three tokens")
	}
	rng := rand.New(rand.NewSource(71))
	losses := make([]float64, 0, epochs)
	for epoch := 0; epoch < epochs; epoch++ {
		var totalLoss float64
		for step := 0; step < stepsPerEpoch; step++ {
			start := rng.Intn(len(ids) - 1)
			end := start + m.Config.Context + 1
			if end > len(ids) {
				end = len(ids)
			}
			window := ids[start:end]
			if len(window) < 2 {
				continue
			}
			cache := m.forward(window[:len(window)-1])
			targets := window[1:]
			totalLoss += meanWindowLoss(cache, targets, nil)
			zeroGradients(m.parameters)
			m.backward(cache, targets, nil)
			m.adamStep(float32(learningRate))
		}
		losses = append(losses, totalLoss/float64(stepsPerEpoch))
	}
	return losses, nil
}

// TrainInstructions trains on prompt-completion pairs using prompt loss masking.
// Gradients are only computed on completion tokens.
func (m *Model) TrainInstructions(examples []InstructionExample, epochs, stepsPerEpoch int, learningRate float64) ([]float64, error) {
	if epochs <= 0 || stepsPerEpoch <= 0 || learningRate <= 0 || math.IsNaN(learningRate) || math.IsInf(learningRate, 0) {
		return nil, fmt.Errorf("epochs, steps, and finite positive learning rate are required")
	}
	if len(examples) == 0 {
		return nil, fmt.Errorf("instruction examples required")
	}
	m.invalidateBackend()

	type preparedExample struct {
		input   []int
		targets []int
		mask    []bool
	}

	var prepared []preparedExample
	for _, ex := range examples {
		promptIDs := m.Tokenizer.Encode(ex.Prompt)
		completionIDs := m.Tokenizer.Encode(" " + strings.TrimSpace(ex.Completion))
		ids := append(promptIDs, completionIDs...)
		if len(ids) < 2 {
			continue
		}
		start := 0
		if len(ids) > m.Config.Context+1 {
			start = len(ids) - (m.Config.Context + 1)
		}
		window := ids[start:]
		promptLen := len(promptIDs) - start
		if promptLen < 0 {
			promptLen = 0
		}
		input := window[:len(window)-1]
		targets := window[1:]
		maskSlice := make([]bool, len(targets))
		hasActive := false
		for i := range targets {
			if i+1 >= promptLen {
				maskSlice[i] = true
				hasActive = true
			}
		}
		if hasActive {
			prepared = append(prepared, preparedExample{input: input, targets: targets, mask: maskSlice})
		}
	}

	if len(prepared) == 0 {
		return nil, fmt.Errorf("no valid instruction examples within context length")
	}

	rng := rand.New(rand.NewSource(71))
	losses := make([]float64, 0, epochs)
	for epoch := 0; epoch < epochs; epoch++ {
		var totalLoss float64
		for step := 0; step < stepsPerEpoch; step++ {
			item := prepared[rng.Intn(len(prepared))]
			cache := m.forward(item.input)
			loss := meanWindowLoss(cache, item.targets, item.mask)
			totalLoss += loss
			zeroGradients(m.parameters)
			m.backward(cache, item.targets, item.mask)
			m.adamStep(float32(learningRate))
		}
		losses = append(losses, totalLoss/float64(stepsPerEpoch))
	}
	return losses, nil
}

// EvaluateFast returns mean next-token cross-entropy like Evaluate but
// scores non-overlapping context-sized blocks in a single forward pass
// per block (~context× fewer forwards). Semantics differ slightly from
// Evaluate: tokens early in each block see only their in-block prefix as
// context rather than a full sliding window. Use for large held-out
// corpora; use Evaluate for exact sliding-window scoring.
func (m *Model) EvaluateFast(texts []string) (float64, error) {
	var ids []int
	for _, text := range texts {
		ids = append(ids, m.Tokenizer.Encode(text)...)
		ids = append(ids, m.Tokenizer.Encode("\n")...)
	}
	if len(ids) < 2 {
		return 0, fmt.Errorf("evaluation corpus must encode to at least two tokens")
	}
	context := m.Config.Context
	if context < 2 {
		context = 2
	}
	var total float64
	var count int64
	vocab := len(m.Tokenizer.Tokens)
	for start := 0; start+1 < len(ids); start += context - 1 {
		end := start + context
		if end > len(ids) {
			end = len(ids)
		}
		w := ids[start:end]
		if len(w) < 2 {
			break
		}
		cache := m.forward(w)
		rows := len(cache.ids)
		for i := 0; i+1 < rows && i+1 < len(w); i++ {
			total += tokenLoss32(cache.logits[i*vocab:(i+1)*vocab], w[i+1])
			count++
		}
	}
	if count == 0 {
		return 0, fmt.Errorf("no scorable tokens in evaluation corpus")
	}
	return total / float64(count), nil
}

// Evaluate returns mean next-token cross-entropy over a held-out corpus.
func (m *Model) Evaluate(texts []string) (float64, error) {
	var ids []int
	for _, text := range texts {
		ids = append(ids, m.Tokenizer.Encode(text)...)
		ids = append(ids, m.Tokenizer.Encode("\n")...)
	}
	if len(ids) < 2 {
		return 0, fmt.Errorf("evaluation corpus must encode to at least two tokens")
	}
	backend := m.inferenceBackend()
	var total float64
	for target := 1; target < len(ids); target++ {
		start := target - m.Config.Context
		if start < 0 {
			start = 0
		}
		total += tokenLoss32(backend.Forward(ids[start:target]), ids[target])
	}
	return total / float64(len(ids)-1), nil
}

func (m *Model) inferenceBackend() inferenceBackend {
	m.backendMu.Lock()
	defer m.backendMu.Unlock()
	if m.backend == nil {
		m.backend = newInferenceBackend(m)
	}
	return m.backend
}

// InferenceBackendName reports the backend of the cached inference
// snapshot ("cpu" or "cuda"), or "unloaded" before the first generation.
// It deliberately does not construct a backend, so a health check cannot
// clone weights or allocate device memory.
func (m *Model) InferenceBackendName() string {
	m.backendMu.Lock()
	defer m.backendMu.Unlock()
	if m.backend == nil {
		return "unloaded"
	}
	return m.backend.Name()
}

// invalidateBackend drops the cached inference snapshot after training has
// changed the weights. Generators already holding a snapshot keep using it.
func (m *Model) invalidateBackend() {
	m.backendMu.Lock()
	m.backend = nil
	m.backendMu.Unlock()
}

func validUTF8Append(prefix []byte, suffix string) bool {
	total := len(prefix) + len(suffix)
	byteAt := func(index int) byte {
		if index < len(prefix) {
			return prefix[index]
		}
		return suffix[index-len(prefix)]
	}
	for i := 0; i < total; {
		first := byteAt(i)
		if first < 0x80 {
			if (first < 0x20 && first != '\n' && first != '\r' && first != '\t') || first == 0x7F {
				return false
			}
			i++
			continue
		}
		length := 0
		switch {
		case first >= 0xC2 && first <= 0xDF:
			length = 2
		case first >= 0xE0 && first <= 0xEF:
			length = 3
		case first >= 0xF0 && first <= 0xF4:
			length = 4
		default:
			return false
		}
		available, check := total-i, length
		incomplete := available < length
		if incomplete {
			check = available
		}
		for offset := 1; offset < check; offset++ {
			b := byteAt(i + offset)
			if b < 0x80 || b > 0xBF {
				return false
			}
			if offset == 1 && ((first == 0xE0 && b < 0xA0) || (first == 0xED && b > 0x9F) || (first == 0xF0 && b < 0x90) || (first == 0xF4 && b > 0x8F)) {
				return false
			}
		}
		if incomplete {
			return true
		}
		i += length
	}
	return true
}

func appendString(prefix []byte, suffix string) []byte {
	for i := 0; i < len(suffix); i++ {
		prefix = append(prefix, suffix[i])
	}
	return prefix
}

// checkpointHeader is the JSON container (format_version 2). All learnable
// parameters are stored as a flat array in the same order that NewModel
// allocates them. Large models are written as safetensors instead; see
// SaveCheckpoint.
// FormatVersion 2 adds training metadata (step, seed, tokenizer fingerprint)
// and optional Adam optimizer state so interrupted runs can resume.
type checkpointHeader struct {
	FormatVersion int               `json:"format_version,omitempty"`
	Config        Config            `json:"config"`
	Tokens        [][]byte          `json:"tokens"`
	Merges        []checkpointMerge `json:"merges"`
	Step          int64             `json:"step,omitempty"`
	Seed          int64             `json:"seed,omitempty"`
	TokenizerHash string            `json:"tokenizer_hash,omitempty"`
	SpecialTokens map[string]int    `json:"special_tokens,omitempty"`
	Values        json.RawMessage   `json:"values"`
	Optimizer     *optimizerRaw     `json:"optimizer,omitempty"`
}

type optimizerRaw struct {
	First  json.RawMessage `json:"first"`
	Second json.RawMessage `json:"second"`
}

type checkpointMerge struct{ Left, Right, Token []byte }

// checkpointTokenizer is the tokenizer part of a checkpoint, shared by the JSON
// and safetensors containers.
type checkpointTokenizer struct {
	Tokens        [][]byte          `json:"tokens"`
	Merges        []checkpointMerge `json:"merges"`
	SpecialTokens map[string]int    `json:"special_tokens,omitempty"`
}

// tokenizerFromCheckpoint rebuilds a tokenizer and restores the well-known
// special tokens when they are not listed explicitly.
func tokenizerFromCheckpoint(tokens [][]byte, merges []checkpointMerge, special map[string]int) *Tokenizer {
	tokenizer := &Tokenizer{
		Tokens:        make([]string, len(tokens)),
		Merges:        make([]Merge, len(merges)),
		SpecialTokens: make(map[string]int, len(special)),
	}
	for k, v := range special {
		tokenizer.SpecialTokens[k] = v
	}
	for i, token := range tokens {
		tokenizer.Tokens[i] = string(token)
		switch tokenizer.Tokens[i] {
		case "<|endoftext|>", "<|im_end|>", "<|im_start|>", "<|pad|>", "<|unk|>":
			if _, ok := tokenizer.SpecialTokens[tokenizer.Tokens[i]]; !ok {
				tokenizer.SpecialTokens[tokenizer.Tokens[i]] = i
			}
		}
	}
	for i, merge := range merges {
		tokenizer.Merges[i] = Merge{Left: string(merge.Left), Right: string(merge.Right), Token: string(merge.Token)}
	}
	tokenizer.reindex()
	return tokenizer
}

// parseFloatSliceInto decodes a JSON array of numbers into target, which must
// have exactly one slot per number. Each value has a fixed index, so splitting
// the work across cores cannot change the result.
func parseFloatSliceInto(raw []byte, target []float32) error {
	raw = bytes.TrimSpace(raw)
	if len(raw) < 2 || raw[0] != '[' || raw[len(raw)-1] != ']' {
		return fmt.Errorf("invalid json float array")
	}
	body := raw[1 : len(raw)-1]
	if workers := runtime.GOMAXPROCS(0); workers > 1 && len(target) >= 1<<15 {
		if ok, err := parseFloatsParallel(body, target, workers); ok {
			return err
		}
		// Chunk element counts did not line up; fall through so the serial
		// parser reports the mismatch.
	}
	n, err := parseFloats(body, target)
	if err != nil {
		return err
	}
	if n != len(target) {
		return fmt.Errorf("checkpoint has %d parameters; model requires %d", n, len(target))
	}
	return nil
}

// parseFloatsParallel cuts body at comma boundaries and parses the chunks
// concurrently into disjoint slices of target. ok is false when the array does
// not hold exactly len(target) numbers, leaving the report to the serial path.
func parseFloatsParallel(body []byte, target []float32, workers int) (ok bool, err error) {
	type chunk struct {
		start, end int
		count      int
		offset     int
	}
	chunks := make([]chunk, 0, workers)
	pos, total := 0, 0
	for pos < len(body) && len(chunks) < workers {
		n := chunk{start: pos, end: len(body)}
		if len(chunks) < workers-1 {
			boundary := len(body) * (len(chunks) + 1) / workers
			if boundary < n.start {
				boundary = n.start
			}
			if comma := bytes.IndexByte(body[boundary:], ','); comma >= 0 {
				n.end = boundary + comma + 1
			}
		}
		// Interior chunks end just after a comma, so they hold exactly one
		// number per comma; only the final chunk holds a trailing number.
		n.count = bytes.Count(body[n.start:n.end], []byte(","))
		if n.end == len(body) {
			n.count++
		}
		n.offset = total
		total += n.count
		chunks = append(chunks, n)
		pos = n.end
	}
	if total != len(target) {
		return false, nil
	}
	errs := make([]error, len(chunks))
	var wg sync.WaitGroup
	for i, c := range chunks {
		wg.Add(1)
		go func(i int, c chunk) {
			defer wg.Done()
			parsed, err := parseFloats(body[c.start:c.end], target[c.offset:c.offset+c.count])
			if err != nil {
				errs[i] = err
			} else if parsed != c.count {
				errs[i] = fmt.Errorf("invalid json float array")
			}
		}(i, c)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return true, err
		}
	}
	return true, nil
}

// parseFloats parses a comma-separated run of JSON numbers into target and
// reports how many it read.
func parseFloats(body []byte, target []float32) (int, error) {
	n := 0
	for i := 0; i < len(body); {
		for i < len(body) && isJSONSeparator(body[i]) {
			i++
		}
		if i >= len(body) {
			break
		}
		start := i
		for i < len(body) && !isJSONSeparator(body[i]) {
			i++
		}
		if n >= len(target) {
			return n, fmt.Errorf("checkpoint has extra parameters beyond model capacity %d", len(target))
		}
		value := unsafe.String(&body[start], i-start)
		v, err := strconv.ParseFloat(value, 32)
		if err != nil {
			return n, fmt.Errorf("parse float %q: %w", value, err)
		}
		target[n] = float32(v)
		n++
	}
	return n, nil
}

func isJSONSeparator(b byte) bool {
	return b == ',' || b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

func skipJSONSpace(data []byte, i int) int {
	for i < len(data) {
		switch data[i] {
		case ' ', '\t', '\n', '\r':
			i++
		default:
			return i
		}
	}
	return i
}

// keyAt returns the offsets of a JSON object key and its value. A token whose
// text equals key never matches, because token strings are followed by a comma
// or a closing bracket rather than a colon.
func keyAt(data []byte, from int, key string) (keyStart, valueStart int, ok bool) {
	needle := []byte(`"` + key + `"`)
	for search := from; search < len(data); {
		idx := bytes.Index(data[search:], needle)
		if idx < 0 {
			return 0, 0, false
		}
		found := search + idx
		value := skipJSONSpace(data, found+len(needle))
		if value < len(data) && data[value] == ':' {
			return found, skipJSONSpace(data, value+1), true
		}
		search = found + 1
	}
	return 0, 0, false
}

// arrayAfterKey returns the byte range of the array value for key. Occurrences
// whose value is not an array (for example a special token that happens to be
// named after the key) are skipped.
func arrayAfterKey(data []byte, from int, key string) (keyStart, start, end int, ok bool) {
	for from < len(data) {
		found, value, matched := keyAt(data, from, key)
		if !matched {
			return 0, 0, 0, false
		}
		if value < len(data) && data[value] == '[' {
			closing := bytes.IndexByte(data[value:], ']')
			if closing < 0 {
				return 0, 0, 0, false
			}
			return found, value, value + closing + 1, true
		}
		from = found + 1
	}
	return 0, 0, 0, false
}

// checkpointArrays holds a checkpoint's header and the raw numeric arrays, so
// weights can be parsed without routing them through encoding/json.
type checkpointArrays struct {
	header       checkpointHeader
	values       []byte
	first        []byte
	second       []byte
	hasOptimizer bool
}

// optimizerArrays returns the Adam moment arrays when the checkpoint carries an
// optimizer object. A present but malformed optimizer is an error, matching the
// strict whole-document decoding path.
func optimizerArrays(data []byte) (first, second []byte, present bool, err error) {
	_, value, ok := keyAt(data, 0, "optimizer")
	if !ok {
		return nil, nil, false, nil
	}
	// Scope the search to the optimizer object, whose only members are the two
	// number arrays, so a token or special-token key named "first" or "second"
	// elsewhere in the file cannot be mistaken for optimizer state.
	end := bytes.IndexByte(data[value:], '}')
	if end < 0 {
		return nil, nil, true, fmt.Errorf("invalid json float array")
	}
	region := data[value : value+end]
	_, firstStart, firstEnd, firstOK := arrayAfterKey(region, 0, "first")
	_, secondStart, secondEnd, secondOK := arrayAfterKey(region, 0, "second")
	if !firstOK || !secondOK {
		return nil, nil, true, fmt.Errorf("invalid json float array")
	}
	return region[firstStart:firstEnd], region[secondStart:secondEnd], true, nil
}

// decodeCheckpoint extracts the header and numeric arrays, preferring a layout
// aware path and falling back to decoding the whole document, so every
// checkpoint that loaded before still loads.
func decodeCheckpoint(data []byte) (checkpointArrays, error) {
	var out checkpointArrays
	if keyStart, start, end, found := arrayAfterKey(data, 0, "values"); found {
		headerJSON := make([]byte, 0, keyStart+1)
		headerJSON = append(headerJSON, bytes.TrimRight(data[:keyStart], " \t\r\n,")...)
		headerJSON = append(headerJSON, '}')
		if err := json.Unmarshal(headerJSON, &out.header); err == nil && len(out.header.Tokens) > 0 {
			first, second, hasOptimizer, err := optimizerArrays(data)
			if err != nil {
				return out, fmt.Errorf("checkpoint optimizer state: %w", err)
			}
			out.values, out.first, out.second, out.hasOptimizer = data[start:end], first, second, hasOptimizer
			return out, nil
		}
	}
	var header checkpointHeader
	if err := json.Unmarshal(data, &header); err != nil {
		return out, err
	}
	out.header = header
	out.values = header.Values
	if header.Optimizer != nil {
		out.first, out.second, out.hasOptimizer = header.Optimizer.First, header.Optimizer.Second, true
	}
	return out, nil
}

// safetensorsCheckpointMinParams is the parameter count at which SaveCheckpoint
// switches containers. Below it a JSON checkpoint is small and easy to inspect
// with ordinary tools; above it the same state is hundreds of megabytes of
// decimal text (a 6M-parameter model with Adam moments is ~370 MB) that has to
// be parsed number by number.
const safetensorsCheckpointMinParams = 1 << 20

// SaveCheckpoint writes tokenizer, weights, and training state (optimizer
// moments, step count, seed) to path.
//
// Models at or above safetensorsCheckpointMinParams are written in safetensors
// format; smaller ones stay JSON. LoadCheckpoint detects either container, so
// both load back the same way. Prefer a .safetensors path for large models: the
// file is binary even when the path ends in .json.
//
// The write is atomic (temporary file, fsync, rename, directory fsync) and is
// refused when the model holds non-finite weights or moments. See
// docs/CHECKPOINTS.md for the durability guarantees and the integrity checks
// LoadCheckpoint applies.
func (m *Model) SaveCheckpoint(path string) error {
	if m.ParameterCount() >= safetensorsCheckpointMinParams {
		return m.SaveSafetensorsCheckpoint(path)
	}
	return m.saveJSONCheckpoint(path)
}

// saveJSONCheckpoint writes the JSON checkpoint container atomically: a
// temporary file, fsync, rename, then a directory fsync (see writeFileAtomic).
// Non-finite weights or moments are refused rather than written.
func (m *Model) saveJSONCheckpoint(path string) error {
	if err := m.validateModelState(); err != nil {
		return fmt.Errorf("refusing to save checkpoint: %w", err)
	}
	return writeFileAtomic(path, 0o600, func(dst io.Writer) error {
		w := bufio.NewWriterSize(dst, 4*1024*1024)
		if err := m.writeJSONCheckpoint(w); err != nil {
			return err
		}
		return w.Flush()
	})
}

// writeJSONCheckpoint streams the format-version-2 JSON container into w.
func (m *Model) writeJSONCheckpoint(w *bufio.Writer) error {
	if _, err := fmt.Fprintf(w, `{"format_version":2,"config":`); err != nil {
		return err
	}
	cfgBytes, err := json.Marshal(m.Config)
	if err != nil {
		return err
	}
	if _, err := w.Write(cfgBytes); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, `,"step":%d,"seed":%d,"tokenizer_hash":%q`, m.step, m.seed, TokenizerFingerprint(m.Tokenizer)); err != nil {
		return err
	}
	if len(m.Tokenizer.SpecialTokens) > 0 {
		stBytes, err := json.Marshal(m.Tokenizer.SpecialTokens)
		if err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, `,"special_tokens":`); err != nil {
			return err
		}
		if _, err := w.Write(stBytes); err != nil {
			return err
		}
	}
	stateTokens := make([][]byte, len(m.Tokenizer.Tokens))
	for i, token := range m.Tokenizer.Tokens {
		stateTokens[i] = []byte(token)
	}
	tokBytes, err := json.Marshal(stateTokens)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, `,"tokens":`); err != nil {
		return err
	}
	if _, err := w.Write(tokBytes); err != nil {
		return err
	}
	merges := make([]checkpointMerge, len(m.Tokenizer.Merges))
	for i, merge := range m.Tokenizer.Merges {
		merges[i] = checkpointMerge{[]byte(merge.Left), []byte(merge.Right), []byte(merge.Token)}
	}
	mgBytes, err := json.Marshal(merges)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, `,"merges":`); err != nil {
		return err
	}
	if _, err := w.Write(mgBytes); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(w, `,"values":[`); err != nil {
		return err
	}
	var buf [64]byte
	firstVal := true
	for _, p := range m.parameters {
		for _, v := range p.values {
			if !firstVal {
				if err := w.WriteByte(','); err != nil {
					return err
				}
			} else {
				firstVal = false
			}
			b := strconv.AppendFloat(buf[:0], float64(v), 'g', -1, 32)
			if _, err := w.Write(b); err != nil {
				return err
			}
		}
	}
	if err := w.WriteByte(']'); err != nil {
		return err
	}

	if m.hasOptimizerMoments() {
		if _, err := fmt.Fprintf(w, `,"optimizer":{"first":[`); err != nil {
			return err
		}
		firstVal = true
		for _, p := range m.parameters {
			for _, v := range p.first {
				if !firstVal {
					if err := w.WriteByte(','); err != nil {
						return err
					}
				} else {
					firstVal = false
				}
				b := strconv.AppendFloat(buf[:0], float64(v), 'g', -1, 32)
				if _, err := w.Write(b); err != nil {
					return err
				}
			}
		}
		if _, err := fmt.Fprintf(w, `],"second":[`); err != nil {
			return err
		}
		firstVal = true
		for _, p := range m.parameters {
			for _, v := range p.second {
				if !firstVal {
					if err := w.WriteByte(','); err != nil {
						return err
					}
				} else {
					firstVal = false
				}
				b := strconv.AppendFloat(buf[:0], float64(v), 'g', -1, 32)
				if _, err := w.Write(b); err != nil {
					return err
				}
			}
		}
		if _, err := fmt.Fprintf(w, `]}`); err != nil {
			return err
		}
	}

	return w.WriteByte('}')
}

// LoadCheckpoint restores model topology, tokenizer, and weights.
func LoadCheckpoint(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if isSafetensorsContainer(data) {
		return loadSafetensorsCheckpoint(data)
	}
	decoded, err := decodeCheckpoint(data)
	if err != nil {
		return nil, fmt.Errorf("decode checkpoint: %w", err)
	}
	header := decoded.header
	if header.FormatVersion < 0 || header.FormatVersion > checkpointFormatVersion {
		return nil, fmt.Errorf("checkpoint format version %d is not supported (this build reads up to %d)",
			header.FormatVersion, checkpointFormatVersion)
	}
	tokenizer := tokenizerFromCheckpoint(header.Tokens, header.Merges, header.SpecialTokens)
	if header.TokenizerHash != "" && TokenizerFingerprint(tokenizer) != header.TokenizerHash {
		return nil, fmt.Errorf("checkpoint tokenizer hash %q does not match serialized tokenizer", header.TokenizerHash)
	}
	model, err := NewModel(header.Config, tokenizer, 1)
	if err != nil {
		return nil, err
	}
	totalParams := model.ParameterCount()
	if len(decoded.values) == 0 {
		return nil, fmt.Errorf("checkpoint has 0 parameters; model requires %d", totalParams)
	}
	flatVals := make([]float32, totalParams)
	if err := parseFloatSliceInto(decoded.values, flatVals); err != nil {
		if hint := legacyLayoutHint(header.Config, len(tokenizer.Tokens), countJSONNumbers(decoded.values)); hint != "" {
			return nil, fmt.Errorf("%w; %s", err, hint)
		}
		return nil, err
	}
	index := 0
	for _, p := range model.parameters {
		copy(p.values, flatVals[index:index+len(p.values)])
		index += len(p.values)
	}
	if header.Step != 0 {
		model.step = header.Step
	}
	if header.Seed != 0 {
		model.seed = header.Seed
	}
	if decoded.hasOptimizer {
		flatFirst := make([]float32, totalParams)
		if err := parseFloatSliceInto(decoded.first, flatFirst); err != nil {
			return nil, fmt.Errorf("checkpoint optimizer state: %w", err)
		}
		flatSecond := make([]float32, totalParams)
		if err := parseFloatSliceInto(decoded.second, flatSecond); err != nil {
			return nil, fmt.Errorf("checkpoint optimizer state: %w", err)
		}
		index = 0
		for _, p := range model.parameters {
			copy(p.first, flatFirst[index:index+len(p.first)])
			copy(p.second, flatSecond[index:index+len(p.second)])
			index += len(p.values)
		}
	}
	if err := model.validateModelState(); err != nil {
		return nil, fmt.Errorf("checkpoint integrity: %w", err)
	}
	return model, nil
}

// Step returns the optimizer step count stored in this model (0 for
// freshly initialized models). Used by model cards and packaging.
func (m *Model) Step() int64 { return m.step }

// ResetStep resets the model step counter to 0, allowing fine-tuning stages
// (such as SFT or DPO) to run their own warmup and decay schedules starting from step 0.
func (m *Model) ResetStep() {
	m.step = 0
}

// hasOptimizerMoments reports whether any Adam moment is non-zero, which is
// what decides whether optimizer state is written to a checkpoint.
func (m *Model) hasOptimizerMoments() bool {
	for _, p := range m.parameters {
		for i := range p.first {
			if p.first[i] != 0 || p.second[i] != 0 {
				return true
			}
		}
	}
	return false
}

// ResetOptimizer clears the Adam first and second moments to zero while
// retaining the learned weights.
func (m *Model) ResetOptimizer() {
	for _, p := range m.parameters {
		for i := range p.first {
			p.first[i] = 0
			p.second[i] = 0
		}
	}
}
