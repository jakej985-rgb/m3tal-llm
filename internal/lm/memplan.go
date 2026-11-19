package lm

import (
	"fmt"
	"log"

	"m3tal-llm/internal/gpu"
	"m3tal-llm/internal/memplan"
)

// inferenceMemoryConfig maps a loaded model's inference shape onto the
// planner's model-independent description. The current backends store fp32
// weights and an fp32 KV cache, so the dtypes are fixed here until
// quantization lands (#1/#63).
func (m *Model) inferenceMemoryConfig() memplan.Config {
	config := m.Config
	config.Normalize()
	return memplan.Config{
		Parameters:      int64(m.ParameterCount()),
		Layers:          config.Layers,
		Width:           config.Width,
		Heads:           config.Heads,
		Vocab:           len(m.Tokenizer.Tokens),
		WeightDType:     memplan.F32,
		KVDType:         memplan.F32,
		ActivationDType: memplan.F32,
	}
}

// inferenceMemoryDecision plans one request at batch 1 and the model's full
// context. It only estimates: no memory is reserved and no allocation changes.
func (m *Model) inferenceMemoryDecision(provider memplan.Provider) (memplan.Decision, error) {
	request := memplan.Request{
		Config:        m.inferenceMemoryConfig(),
		Batch:         1,
		ContextTokens: m.Config.Context,
	}
	return memplan.Plan(request, memplan.Limits{Provider: provider})
}

// gpuMemoryProvider adapts the CUDA free-memory query to memplan.Provider.
// Any query failure (no CUDA build, no device, driver error) is reported as
// ErrUnsupported so the plan stays advisory instead of refusing work.
type gpuMemoryProvider struct{}

func (gpuMemoryProvider) AvailableBytes() (int64, error) {
	free, err := gpu.AvailableMemoryBytes()
	if err != nil {
		return 0, fmt.Errorf("%w: %v", memplan.ErrUnsupported, err)
	}
	return free, nil
}

// logInferenceMemoryPlan estimates what the selected backend is about to
// allocate and logs it. This is a read-only step: the plan neither reserves
// nor reduces anything, and allocation behavior is unchanged. A shortfall is
// logged so the estimates can be calibrated before enforcement.
func logInferenceMemoryPlan(m *Model, backend string, provider memplan.Provider) {
	decision, err := m.inferenceMemoryDecision(provider)
	if err != nil {
		log.Printf("memplan: backend=%s source=%s requested=%s admitted=%s error=%v",
			backend, decision.Source, decision.RequestedRequirements, decision.Requirements, err)
		return
	}
	log.Printf("memplan: backend=%s source=%s available=%d admitted=batch:%d/context:%d advisory=%t %s",
		backend, decision.Source, decision.AvailableBytes,
		decision.Admitted.Batch, decision.Admitted.ContextTokens, decision.Advisory,
		decision.Requirements)
}
