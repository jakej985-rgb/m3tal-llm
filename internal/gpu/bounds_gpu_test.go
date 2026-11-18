//go:build cuda && cgo && amd64

package gpu

// Device-backed boundary tests. They skip cleanly when no usable CUDA device
// is present, so the same binary is safe on CPU-only machines. Validation
// coverage that needs no device lives in bounds_test.go.

import (
	"errors"
	"testing"
)

func TestDeviceBoundsRejectOutOfRange(t *testing.T) {
	requireGPU(t)

	a, err := AllocF32(4)
	if err != nil {
		t.Fatalf("AllocF32: %v", err)
	}
	defer a.Free()
	b, err := AllocF32(4)
	if err != nil {
		t.Fatalf("AllocF32: %v", err)
	}
	defer b.Free()
	out, err := AllocF32(4)
	if err != nil {
		t.Fatalf("AllocF32: %v", err)
	}
	defer out.Free()

	if err := a.Upload([]float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("Upload: %v", err)
	}
	if err := b.Upload([]float32{1, 0, 0, 1}); err != nil {
		t.Fatalf("Upload: %v", err)
	}

	// Partial transfers are rejected instead of leaving device memory stale.
	if err := a.Upload(make([]float32, 8)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("oversized upload: got %v, want ErrInvalid", err)
	}
	if err := a.Download(make([]float32, 2)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("undersized download: got %v, want ErrInvalid", err)
	}
	if err := a.UploadBytes(make([]byte, 15)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("short byte upload: got %v, want ErrInvalid", err)
	}

	// Device-to-device copies: valid span, out-of-range, negative, overlap.
	if err := MemcpyD2D(out, 0, a, 0, 16); err != nil {
		t.Fatalf("MemcpyD2D: %v", err)
	}
	if err := MemcpyD2D(out, 8, a, 0, 16); !errors.Is(err, ErrInvalid) {
		t.Fatalf("out-of-range d2d: got %v, want ErrInvalid", err)
	}
	if err := MemcpyD2D(out, -1, a, 0, 4); !errors.Is(err, ErrInvalid) {
		t.Fatalf("negative offset d2d: got %v, want ErrInvalid", err)
	}
	if err := MemcpyD2D(a, 4, a, 0, 8); !errors.Is(err, ErrInvalid) {
		t.Fatalf("overlapping d2d: got %v, want ErrInvalid", err)
	}
	if err := MemcpyD2D(a, 0, a, 0, 16); err != nil {
		t.Fatalf("identity d2d copy: %v", err)
	}

	// Freed buffers are rejected even when the size would fit.
	freed, err := AllocF32(4)
	if err != nil {
		t.Fatalf("AllocF32: %v", err)
	}
	freed.Free()
	if err := freed.Upload(make([]float32, 4)); !errors.Is(err, ErrInvalid) {
		t.Fatalf("upload into freed buffer: got %v, want ErrInvalid", err)
	}
	if err := freed.Zero(); !errors.Is(err, ErrInvalid) {
		t.Fatalf("zero on freed buffer: got %v, want ErrInvalid", err)
	}
	if err := GemmF32(a, freed, out, 2, 2, 2); !errors.Is(err, ErrInvalid) {
		t.Fatalf("gemm with freed operand: got %v, want ErrInvalid", err)
	}

	// A buffer from another device is rejected before the launch.
	other := &Buffer{bytes: 16, device: a.device + 1}
	if err := GemmF32(a, other, out, 2, 2, 2); !errors.Is(err, ErrInvalid) {
		t.Fatalf("gemm across devices: got %v, want ErrInvalid", err)
	}

	// Empty buffers are legal and their operations are no-ops.
	empty, err := AllocF32(0)
	if err != nil {
		t.Fatalf("AllocF32(0): %v", err)
	}
	defer empty.Free()
	if err := empty.Upload(nil); err != nil {
		t.Fatalf("Upload(nil) on empty buffer: %v", err)
	}
	if err := empty.Zero(); err != nil {
		t.Fatalf("Zero on empty buffer: %v", err)
	}

	// The valid 2x2 gemm still runs: out = [[1,2],[3,4]] * I.
	if err := out.Zero(); err != nil {
		t.Fatalf("Zero: %v", err)
	}
	if err := GemmF32(a, b, out, 2, 2, 2); err != nil {
		t.Fatalf("GemmF32: %v", err)
	}
	got := make([]float32, 4)
	if err := out.Download(got); err != nil {
		t.Fatalf("Download: %v", err)
	}
	want := []float32{1, 2, 3, 4}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("gemm result %v, want %v", got, want)
		}
	}
}

func TestDeviceAllocRejectsInvalidSizes(t *testing.T) {
	requireGPU(t)

	if _, err := AllocF32(-1); !errors.Is(err, ErrInvalid) {
		t.Fatalf("AllocF32(-1): got %v, want ErrInvalid", err)
	}
	if _, err := AllocF64(-1); !errors.Is(err, ErrInvalid) {
		t.Fatalf("AllocF64(-1): got %v, want ErrInvalid", err)
	}
	if _, err := AllocBytes(-1); !errors.Is(err, ErrInvalid) {
		t.Fatalf("AllocBytes(-1): got %v, want ErrInvalid", err)
	}
	if _, err := AllocF32(int(^uint(0)>>1) / 2); !errors.Is(err, ErrInvalid) {
		t.Fatalf("overflowing allocation: got %v, want ErrInvalid", err)
	}
}
