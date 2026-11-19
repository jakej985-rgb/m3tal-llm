//go:build cuda

package lm

import (
	"testing"

	"m3tal-llm/internal/memplan"
)

// TestCUDABackendAllocationMatchesPlan checks the device upload plus the
// retained host fallback clone against the planner's estimate.
func TestCUDABackendAllocationMatchesPlan(t *testing.T) {
	model := newTestModel(t)
	backend := newCUDAOrSkip(t, model)
	decision, err := model.inferenceMemoryDecision(gpuMemoryProvider{})
	if err != nil {
		t.Fatal(err)
	}
	if decision.Advisory {
		t.Fatal("expected a device budget on a usable CUDA device")
	}

	weights, host := backend.AllocatedBytes()
	if weights != decision.Requirements.WeightsBytes {
		t.Fatalf("device weights = %d, want estimate %d", weights, decision.Requirements.WeightsBytes)
	}
	if host != decision.Requirements.WeightsBytes {
		t.Fatalf("host clone = %d, want one fp32 copy %d", host, decision.Requirements.WeightsBytes)
	}
	comparison := memplan.Compare(decision.Requirements, memplan.ActualAllocation{
		WeightsBytes: weights,
		HostBytes:    host,
	})
	if comparison.WeightDeltaBytes != 0 {
		t.Fatalf("weight delta = %d, want 0", comparison.WeightDeltaBytes)
	}
}
