# 03 Model Architecture Scaling

## Goal
Scale the pure-Go decoder-only Transformer from 0.9M to **6.05M parameters** to achieve sufficient neural capacity for coherent grammar, semantic reasoning, and multi-turn context retention while maintaining sub-50ms CPU inference and a memory footprint under 30 MB.

## Architectural Configuration

```text
Config {
    Context: 256 tokens    (holds ~180–200 words at a time)
    Width:   256           (hidden dimension d_model)
    Layers:  8             (depth of abstract reasoning)
    Heads:   8             (32 dimensions per attention head)
    Vocab:   1536 tokens   (calibrated BPE vocabulary)
}
```

### Component Parameter Breakdown

| Component | Matrix Dimensions | Parameters |
| :--- | :--- | :--- |
| **Token Embedding** | $V \times d_{\text{model}} = 1536 \times 256$ | 393,216 (0.39M) |
| **RMSNorm Gammas** | $2 \times d_{\text{model}} \times 8$ | 4,096 (<0.01M) |
| **Attention Blocks** | $8 \text{ layers} \times 4 \times (W^2 + W)$ | 2,099,200 (2.10M) |
| **SwiGLU FFN Blocks** | $8 \text{ layers} \times (6 W^2 + 5 W)$ | 3,153,920 (3.15M) |
| **Output Projection** | $d_{\text{model}} \times V + V$ | 394,752 (0.39M) |
| **Total Model Weights** | | **6,053,376 (6.05M)** |

### Resource Budgets (FP32)
- **Inference Weights**: **24.2 MB**
- **Training State (Weights + Gradients + 2 Adam Moments)**: **96.8 MB**
- **Inference Latency (CPU with SIMD AVX2)**: **~15–25 ms per token**
- **VRAM Required**: 0 MB (100% CPU host memory; optional CUDA acceleration supported)

## Key Transformer Mechanisms in Pure Go
1. **Rotary Position Embeddings (RoPE)**: Applied to query/key projections in `internal/lm/model.go`. Allows relative position tracking up to context length without static learned tables.
2. **RMSNorm**: Applied before self-attention and before feed-forward blocks for numerical stability.
3. **SwiGLU Activation**: Gated feed-forward network ($W_{\text{gate}} \times \text{Swish} \odot W_{\text{up}}$ followed by $W_{\text{down}}$) providing non-linear representation capacity superior to traditional ReLU/GELU.

## Definition of Done
- Model config passes validation in `m3tal-lm model-info`.
- Matrix allocation tests pass without memory leaks or race conditions.
- SIMD AVX2 kernel benchmarks verified in `internal/tensor/kernel`.
