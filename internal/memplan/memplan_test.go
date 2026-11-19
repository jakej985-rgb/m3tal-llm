package memplan

import (
	"errors"
	"math"
	"strings"
	"testing"
)

// fakeProvider stands in for a device memory query in tests.
type fakeProvider struct {
	available int64
	err       error
}

func (f *fakeProvider) AvailableBytes() (int64, error) {
	return f.available, f.err
}

// testConfig has hand-checkable numbers: 1000 params, 2 layers, width 8,
// 2 heads, vocab 16, fp32, no headroom.
//
//	weights  1000*4                                      = 4000
//	kv       2*2 layers*1*4 tokens*8 width*4             =  512
//	scratch  (1*4*8*28*2 + 2*1*4*16 + 1*2*4*4)*4        = 7808
//	total                                                = 12320
func testConfig() Config {
	return Config{
		Parameters:       1000,
		Layers:           2,
		Width:            8,
		Heads:            2,
		Vocab:            16,
		WeightDType:      F32,
		KVDType:          F32,
		ActivationDType:  F32,
		HeadroomFraction: -1, // disabled for exact arithmetic
	}
}

func testRequest(batch, context int) Request {
	cfg := testConfig()
	return Request{Config: cfg, Batch: batch, ContextTokens: context}
}

func TestCompareReportsWeightDelta(t *testing.T) {
	estimated := Requirements{
		WeightsBytes:  4000,
		KVCacheBytes:  512,
		ScratchBytes:  3456,
		HeadroomBytes: 797,
		TotalBytes:    8765,
	}
	exact := Compare(estimated, ActualAllocation{WeightsBytes: 4000, HostBytes: 1000})
	if exact.WeightDeltaBytes != 0 {
		t.Fatalf("weight delta = %d, want 0", exact.WeightDeltaBytes)
	}
	if exact.HostOverheadBytes != 1000 {
		t.Fatalf("host overhead = %d, want 1000", exact.HostOverheadBytes)
	}
	if exact.Actual.ConstructionBytes() != 5000 {
		t.Fatalf("construction bytes = %d, want 5000", exact.Actual.ConstructionBytes())
	}
	if got := exact.String(); !strings.Contains(got, "weight_delta=0B") || !strings.Contains(got, "host_copy=1000B") {
		t.Fatalf("comparison string missing fields: %s", got)
	}

	drifted := Compare(estimated, ActualAllocation{WeightsBytes: 4096})
	if drifted.WeightDeltaBytes != 96 {
		t.Fatalf("drifted weight delta = %d, want 96", drifted.WeightDeltaBytes)
	}
}

func TestEstimateComponents(t *testing.T) {
	got, err := Estimate(testRequest(1, 4))
	if err != nil {
		t.Fatal(err)
	}
	want := Requirements{
		WeightsBytes: 4000,
		KVCacheBytes: 512,
		ScratchBytes: 7808,
		TotalBytes:   12320,
	}
	if got != want {
		t.Fatalf("Estimate = %+v, want %+v", got, want)
	}
}

func TestEstimateHeadroom(t *testing.T) {
	cfg := testConfig()
	cfg.HeadroomFraction = 0.10
	got, err := Estimate(Request{Config: cfg, Batch: 1, ContextTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	if got.HeadroomBytes != 1232 { // ceil(12320 * 0.10)
		t.Fatalf("headroom = %d, want 1232", got.HeadroomBytes)
	}
	if got.TotalBytes != 13552 {
		t.Fatalf("total = %d, want 13552", got.TotalBytes)
	}
}

func TestEstimateDefaultsHeadroomWhenUnset(t *testing.T) {
	cfg := testConfig()
	cfg.HeadroomFraction = 0 // select the default 10%
	got, err := Estimate(Request{Config: cfg, Batch: 1, ContextTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	if got.HeadroomBytes != 1232 {
		t.Fatalf("default headroom = %d, want 1232", got.HeadroomBytes)
	}
}

func TestEstimateQuantizedWeights(t *testing.T) {
	tests := []struct {
		name         string
		dtype        DType
		group        int
		zeroPointBit int
		wantWeights  int64
		wantMetadata int64
	}{
		// 1000 int8 weights = 1000 bytes; 8 groups (ceil 1000/128) of fp16
		// scale = 16 bytes.
		{name: "int8 grouped", dtype: INT8, group: 128, wantWeights: 1000, wantMetadata: 16},
		// 1000 int4 weights = 500 bytes; same 8-group metadata.
		{name: "int4 grouped", dtype: INT4, group: 128, wantWeights: 500, wantMetadata: 16},
		// Asymmetric: 4 extra zero-point bits per group -> 20 bits * 8 = 20 bytes.
		{name: "int4 asymmetric", dtype: INT4, group: 128, zeroPointBit: 4, wantWeights: 500, wantMetadata: 20},
		// With no group size, INT8 uses one per-tensor scale (2 bytes).
		{name: "int8 per-tensor", dtype: INT8, group: 0, wantWeights: 1000, wantMetadata: 2},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := testConfig()
			cfg.WeightDType = tc.dtype
			cfg.QuantGroupSize = tc.group
			cfg.ZeroPointBits = tc.zeroPointBit
			got, err := Estimate(Request{Config: cfg, Batch: 1, ContextTokens: 4})
			if err != nil {
				t.Fatal(err)
			}
			if got.WeightsBytes != tc.wantWeights {
				t.Fatalf("weights = %d, want %d", got.WeightsBytes, tc.wantWeights)
			}
			if got.QuantMetadataBytes != tc.wantMetadata {
				t.Fatalf("metadata = %d, want %d", got.QuantMetadataBytes, tc.wantMetadata)
			}
		})
	}
}

func TestEstimateFloatWeightsHaveNoQuantMetadata(t *testing.T) {
	cfg := testConfig()
	cfg.QuantGroupSize = 128
	cfg.WeightDType = F16
	got, err := Estimate(Request{Config: cfg, Batch: 1, ContextTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	if got.QuantMetadataBytes != 0 {
		t.Fatalf("metadata = %d, want 0 for float weights", got.QuantMetadataBytes)
	}
	if got.WeightsBytes != 2000 {
		t.Fatalf("weights = %d, want 2000", got.WeightsBytes)
	}
}

func TestEstimateRejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name   string
		req    Request
		target error
	}{
		{name: "zero layers", req: Request{Config: Config{Parameters: 1, Width: 8, Heads: 2, Vocab: 16}, Batch: 1, ContextTokens: 1}, target: ErrInvalidConfig},
		{name: "unknown weight dtype", req: Request{Config: Config{Parameters: 1, Layers: 1, Width: 8, Heads: 2, Vocab: 16, WeightDType: DType(99)}, Batch: 1, ContextTokens: 1}, target: ErrInvalidConfig},
		{name: "zero batch", req: Request{Config: testConfig(), Batch: 0, ContextTokens: 1}, target: ErrInvalidConfig},
		{name: "zero context", req: Request{Config: testConfig(), Batch: 1, ContextTokens: 0}, target: ErrInvalidConfig},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Estimate(tc.req); !errors.Is(err, tc.target) {
				t.Fatalf("err = %v, want %v", err, tc.target)
			}
		})
	}
}

func TestEstimateOverflow(t *testing.T) {
	cfg := testConfig()
	cfg.Parameters = math.MaxInt64
	_, err := Estimate(Request{Config: cfg, Batch: 1, ContextTokens: 1})
	if !errors.Is(err, ErrOverflow) {
		t.Fatalf("err = %v, want ErrOverflow", err)
	}
}

func TestEstimateKVScalesWithContext(t *testing.T) {
	short, err := Estimate(testRequest(1, 4))
	if err != nil {
		t.Fatal(err)
	}
	long, err := Estimate(testRequest(1, 8))
	if err != nil {
		t.Fatal(err)
	}
	if long.KVCacheBytes != 2*short.KVCacheBytes {
		t.Fatalf("kv at ctx8 = %d, want 2x ctx4 (%d)", long.KVCacheBytes, short.KVCacheBytes)
	}
}

func TestPlanFitsAsRequested(t *testing.T) {
	req := testRequest(1, 4)
	decision, err := Plan(req, Limits{Provider: &fakeProvider{available: 12320}})
	if err != nil {
		t.Fatal(err)
	}
	if decision.ReducedBatch || decision.ReducedContext || decision.Advisory {
		t.Fatalf("unexpected reduction/advisory: %+v", decision)
	}
	if decision.Source != SourceProvider {
		t.Fatalf("source = %q, want provider", decision.Source)
	}
	if decision.Admitted != req {
		t.Fatalf("admitted = %+v, want requested", decision.Admitted)
	}
	if decision.AvailableBytes != 12320 {
		t.Fatalf("available = %d, want 12320", decision.AvailableBytes)
	}
}

func TestPlanBoundaryOneByteShortReducesContext(t *testing.T) {
	decision, err := Plan(testRequest(1, 4), Limits{Provider: &fakeProvider{available: 12319}})
	if err != nil {
		t.Fatal(err)
	}
	if !decision.ReducedContext {
		t.Fatal("expected the context to be reduced by one byte short")
	}
	if decision.ReducedBatch {
		t.Fatal("batch should not change for a batch-1 request")
	}
	if decision.Admitted.ContextTokens != 3 {
		t.Fatalf("admitted context = %d, want 3 (largest that fits)", decision.Admitted.ContextTokens)
	}
	if decision.Requirements.TotalBytes > decision.AvailableBytes {
		t.Fatalf("admitted %d bytes > available %d", decision.Requirements.TotalBytes, decision.AvailableBytes)
	}
	if decision.RequestedRequirements.TotalBytes != 12320 {
		t.Fatalf("requested requirements = %d, want 12320 preserved", decision.RequestedRequirements.TotalBytes)
	}
}

func TestPlanReducesBatchBeforeContext(t *testing.T) {
	req := testRequest(2, 4) // 20640 bytes
	decision, err := Plan(req, Limits{Provider: &fakeProvider{available: 13000}})
	if err != nil {
		t.Fatal(err)
	}
	if !decision.ReducedBatch {
		t.Fatal("expected batch reduction")
	}
	if decision.ReducedContext {
		t.Fatal("context should stay at 4 when batch 1 fits")
	}
	if decision.Admitted.Batch != 1 || decision.Admitted.ContextTokens != 4 {
		t.Fatalf("admitted = %+v, want batch 1 context 4", decision.Admitted)
	}
	if decision.Requirements.TotalBytes != 12320 {
		t.Fatalf("admitted requirements = %d, want 12320", decision.Requirements.TotalBytes)
	}
}

func TestPlanLowMemoryRefuses(t *testing.T) {
	decision, err := Plan(testRequest(1, 4), Limits{Provider: &fakeProvider{available: 1000}})
	if !errors.Is(err, ErrInsufficientMemory) {
		t.Fatalf("err = %v, want ErrInsufficientMemory", err)
	}
	if decision.Admitted.ContextTokens != 1 {
		t.Fatalf("admitted context = %d, want the minimum of 1", decision.Admitted.ContextTokens)
	}
	if decision.Requirements.TotalBytes != 6056 {
		t.Fatalf("minimum requirements = %d, want 6056", decision.Requirements.TotalBytes)
	}
	if decision.AvailableBytes != 1000 {
		t.Fatalf("available = %d, want 1000", decision.AvailableBytes)
	}
}

func TestPlanHonorsMinimumContext(t *testing.T) {
	// Context 2 fits in 8500 bytes, but the caller forbids going below 3.
	decision, err := Plan(testRequest(1, 4), Limits{
		Provider:         &fakeProvider{available: 8500},
		MinContextTokens: 3,
	})
	if !errors.Is(err, ErrInsufficientMemory) {
		t.Fatalf("err = %v, want ErrInsufficientMemory", err)
	}
	if decision.Admitted.ContextTokens != 3 {
		t.Fatalf("admitted context = %d, want 3", decision.Admitted.ContextTokens)
	}
	if decision.Requirements.TotalBytes != 10216 {
		t.Fatalf("requirements = %d, want 10216", decision.Requirements.TotalBytes)
	}
}

func TestPlanAdvisoryWhenProviderUnsupported(t *testing.T) {
	decision, err := Plan(testRequest(2, 8), Limits{
		Provider: &fakeProvider{err: ErrUnsupported},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !decision.Advisory {
		t.Fatal("expected an advisory plan when availability is unknown")
	}
	if decision.Source != SourceUnknown {
		t.Fatalf("source = %q, want unknown", decision.Source)
	}
	if decision.AvailableBytes != -1 {
		t.Fatalf("available = %d, want -1", decision.AvailableBytes)
	}
	if decision.Admitted != decision.Requested {
		t.Fatal("advisory plan must not reduce the request")
	}
	if decision.Requirements != decision.RequestedRequirements {
		t.Fatal("advisory requirements must match the request")
	}
	if !errors.Is(decision.ProviderError, ErrUnsupported) {
		t.Fatalf("provider error = %v, want ErrUnsupported", decision.ProviderError)
	}
}

func TestPlanFallsBackToUserCapOnProviderError(t *testing.T) {
	req := testRequest(2, 4) // 20640 bytes
	decision, err := Plan(req, Limits{
		Provider:     &fakeProvider{err: ErrUnsupported},
		UserCapBytes: 13000,
	})
	if err != nil {
		t.Fatal(err)
	}
	if decision.Advisory {
		t.Fatal("a user cap must keep the plan enforceable")
	}
	if decision.Source != SourceUserCap {
		t.Fatalf("source = %q, want user_cap", decision.Source)
	}
	if !decision.ReducedBatch || decision.Admitted.Batch != 1 {
		t.Fatalf("admitted = %+v, want a batch-1 plan", decision.Admitted)
	}
}

func TestPlanTreatsNegativeAvailabilityAsUnknown(t *testing.T) {
	decision, err := Plan(testRequest(1, 4), Limits{Provider: &fakeProvider{available: -5}})
	if err != nil {
		t.Fatal(err)
	}
	if !decision.Advisory {
		t.Fatal("expected advisory fallback for an invalid provider value")
	}
	if decision.ProviderError == nil {
		t.Fatal("expected the invalid availability to be recorded")
	}
}

func TestPlanUsesStrictestBudget(t *testing.T) {
	t.Run("user cap is stricter", func(t *testing.T) {
		decision, err := Plan(testRequest(2, 4), Limits{
			Provider:     &fakeProvider{available: 100000},
			UserCapBytes: 13000,
		})
		if err != nil {
			t.Fatal(err)
		}
		if decision.Source != SourceUserCap || decision.AvailableBytes != 13000 {
			t.Fatalf("source=%q available=%d, want user_cap/13000", decision.Source, decision.AvailableBytes)
		}
	})
	t.Run("provider is stricter", func(t *testing.T) {
		decision, err := Plan(testRequest(2, 4), Limits{
			Provider:     &fakeProvider{available: 13000},
			UserCapBytes: 100000,
		})
		if err != nil {
			t.Fatal(err)
		}
		if decision.Source != SourceProvider || decision.AvailableBytes != 13000 {
			t.Fatalf("source=%q available=%d, want provider/13000", decision.Source, decision.AvailableBytes)
		}
	})
}
