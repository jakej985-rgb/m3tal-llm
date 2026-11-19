package lm

import (
	"errors"
	"testing"

	"m3tal-llm/internal/memplan"
)

// TestInferenceMemoryConfigMapsModel checks the lm.Config -> memplan.Config
// mapping uses the model's real parameter count and inference shape.
func TestInferenceMemoryConfigMapsModel(t *testing.T) {
	tok, err := TrainTokenizer([]string{"hello world hello there"}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 16, Width: 16, Layers: 2, Heads: 4}, tok, 1)
	if err != nil {
		t.Fatal(err)
	}

	cfg := model.inferenceMemoryConfig()
	if cfg.Parameters != int64(model.ParameterCount()) {
		t.Fatalf("parameters = %d, want %d", cfg.Parameters, model.ParameterCount())
	}
	if cfg.Layers != 2 || cfg.Width != 16 || cfg.Heads != 4 {
		t.Fatalf("shape = layers:%d width:%d heads:%d, want 2/16/4", cfg.Layers, cfg.Width, cfg.Heads)
	}
	if cfg.Vocab != len(tok.Tokens) {
		t.Fatalf("vocab = %d, want %d", cfg.Vocab, len(tok.Tokens))
	}
	if cfg.WeightDType != memplan.F32 || cfg.KVDType != memplan.F32 || cfg.ActivationDType != memplan.F32 {
		t.Fatalf("dtypes = %s/%s/%s, want f32/f32/f32", cfg.WeightDType, cfg.KVDType, cfg.ActivationDType)
	}

	requirements, err := memplan.Estimate(memplan.Request{
		Config:        cfg,
		Batch:         1,
		ContextTokens: model.Config.Context,
	})
	if err != nil {
		t.Fatal(err)
	}
	if requirements.WeightsBytes != int64(model.ParameterCount())*4 {
		t.Fatalf("weight estimate = %d, want %d", requirements.WeightsBytes, int64(model.ParameterCount())*4)
	}
	if requirements.KVCacheBytes == 0 || requirements.ScratchBytes == 0 {
		t.Fatalf("expected non-zero KV and scratch estimates, got %s", requirements)
	}
}

// TestInferenceMemoryDecisionWithoutProviderIsAdvisory checks the CPU path:
// without a device query the plan reports unknown availability and refuses
// nothing, so wiring it in cannot change behavior yet.
func TestInferenceMemoryDecisionWithoutProviderIsAdvisory(t *testing.T) {
	tok, err := TrainTokenizer([]string{"hello world hello there"}, 256)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewModel(Config{Context: 16, Width: 16, Layers: 1, Heads: 2}, tok, 1)
	if err != nil {
		t.Fatal(err)
	}
	decision, err := model.inferenceMemoryDecision(nil)
	if err != nil {
		t.Fatal(err)
	}
	if !decision.Advisory {
		t.Fatal("expected an advisory plan without a provider")
	}
	if decision.Source != memplan.SourceUnknown {
		t.Fatalf("source = %q, want unknown", decision.Source)
	}
	if decision.ReducedBatch || decision.ReducedContext {
		t.Fatal("advisory plan must not reduce the request")
	}
	if decision.Requirements.TotalBytes == 0 {
		t.Fatal("expected a non-zero total estimate")
	}
}

// TestGPUProviderReportsUnsupportedWithoutCUDA checks the non-CUDA build
// adapter: the missing device query becomes ErrUnsupported, which the planner
// treats as an unknown budget rather than a refusal.
func TestGPUProviderReportsUnsupportedWithoutCUDA(t *testing.T) {
	_, err := gpuMemoryProvider{}.AvailableBytes()
	if err == nil {
		t.Skip("CUDA memory query succeeded; non-CUDA fallback not exercised")
	}
	if !errors.Is(err, memplan.ErrUnsupported) {
		t.Fatalf("provider error = %v, want ErrUnsupported", err)
	}
}
