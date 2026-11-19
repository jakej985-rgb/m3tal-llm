//go:build !cuda

package lm

// newInferenceBackend selects the inference backend. Without the cuda
// build tag only the CPU backend exists.
func newInferenceBackend(m *Model) inferenceBackend {
	// Read-only planning step: log the estimate before the CPU snapshot is
	// cloned. Availability is unknown here, so the plan is advisory.
	logInferenceMemoryPlan(m, "cpu", nil)
	return newCPUInferenceBackend(m)
}
