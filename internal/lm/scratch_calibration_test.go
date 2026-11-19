package lm

import (
	"runtime"
	"testing"
)

// measureAllocBytes reports the cumulative bytes allocated by fn. It is a
// real measurement (runtime.MemStats), not an analytic estimate: the CPU
// forward paths allocate every scratch buffer with make(), so the delta
// captures the buffers the planner is trying to model.
func measureAllocBytes(fn func()) uint64 {
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	fn()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

// TestScratchEstimateCoversMeasuredAllocations is the calibration guard for
// memplan.DefaultActivationBuffers. It measures real allocations of the CPU
// batch-forward and KV generation paths and requires the planner's scratch
// estimate to cover them, so the estimate cannot drift below reality without
// failing here.
func TestScratchEstimateCoversMeasuredAllocations(t *testing.T) {
	// Single P keeps parallelFor serial, so TotalAlloc reflects the model's
	// buffers instead of transient goroutine stacks.
	previous := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(previous)

	const context = 32
	for _, layers := range []int{1, 2, 4} {
		tok, err := TrainTokenizer([]string{"hello world hello there calibration corpus"}, 256)
		if err != nil {
			t.Fatal(err)
		}
		model, err := NewModel(Config{Context: context, Width: 32, Layers: layers, Heads: 4}, tok, 1)
		if err != nil {
			t.Fatal(err)
		}
		backend := newCPUInferenceBackend(model)
		ids := make([]int, context)
		for i := range ids {
			ids[i] = i % len(tok.Tokens)
		}
		decision, err := model.inferenceMemoryDecision(nil)
		if err != nil {
			t.Fatal(err)
		}
		scratch := decision.Requirements.ScratchBytes

		forward := measureAllocBytes(func() { backend.Forward(ids) })

		prefix := backend.NewKVCache()
		for i, id := range ids {
			backend.ForwardStepKV(id, i, prefix)
		}
		decode := measureAllocBytes(func() { backend.ForwardStepKV(0, len(ids), prefix) })
		prefix.Free()

		prefill := measureAllocBytes(func() {
			cache := backend.NewKVCache()
			for i, id := range ids {
				backend.ForwardStepKV(id, i, cache)
			}
			cache.Free()
		})
		kv := uint64(decision.Requirements.KVCacheBytes)

		t.Logf("layers=%d vocab=%d scratch_estimate=%dB forward=%dB prefill=%dB kv=%dB decode_step=%dB",
			layers, len(tok.Tokens), scratch, forward, prefill, kv, decode)

		if forward > uint64(scratch) {
			t.Fatalf("layers=%d batch forward allocated %d bytes, estimate only %d", layers, forward, scratch)
		}
		if decode > uint64(scratch) {
			t.Fatalf("layers=%d decode step allocated %d bytes, estimate only %d", layers, decode, scratch)
		}
		if prefill > uint64(scratch)+kv {
			t.Fatalf("layers=%d prefill allocated %d bytes, estimate scratch+kv is %d", layers, prefill, uint64(scratch)+kv)
		}
	}
}
