package gpu

// Boundary-validation tests. They call the shared validators and only the
// operation paths that fail validation, so they never touch a device and run
// on every build, with or without CUDA. Device-backed behaviour is covered by
// bounds_gpu_test.go behind the cuda build tag.

import (
	"errors"
	"math"
	"strings"
	"testing"
)

// sized returns a buffer placeholder with the given allocation size and
// device for validation-only tests. No device call is ever made with it.
func sized(bytes, device int) *Buffer {
	return &Buffer{bytes: bytes, device: device}
}

// freedSized returns a buffer placeholder that has already been freed.
func freedSized(bytes int) *Buffer {
	b := sized(bytes, 0)
	b.freed = true
	return b
}

func TestElemBytes(t *testing.T) {
	tests := []struct {
		name    string
		elems   int
		size    int
		want    int
		wantErr string
	}{
		{name: "zero is an empty allocation", elems: 0, size: 4, want: 0},
		{name: "normal", elems: 3, size: 4, want: 12},
		{name: "float64", elems: 3, size: 8, want: 24},
		{name: "negative count", elems: -1, size: 4, wantErr: "negative element count"},
		{name: "negative size", elems: 1, size: -4, wantErr: "invalid element size"},
		{name: "overflow", elems: math.MaxInt/4 + 1, size: 4, wantErr: "overflow"},
		{name: "max int overflows", elems: math.MaxInt, size: 4, wantErr: "overflow"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := elemBytes(tc.elems, tc.size, "test")
			if tc.wantErr != "" {
				if !errors.Is(err, ErrInvalid) {
					t.Fatalf("want ErrInvalid, got %v", err)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("error %q does not mention %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %d bytes, want %d", got, tc.want)
			}
		})
	}
}

func TestDimBytes(t *testing.T) {
	tests := []struct {
		name    string
		size    int
		dims    []int
		want    int
		wantErr string
	}{
		{name: "gemm operand", size: 4, dims: []int{4, 4}, want: 64},
		{name: "attention tensor", size: 4, dims: []int{2, 2, 3}, want: 48},
		{name: "zero dimension", size: 4, dims: []int{4, 0}, wantErr: "non-positive dimension"},
		{name: "negative dimension", size: 4, dims: []int{-1}, wantErr: "non-positive dimension"},
		{name: "product overflows", size: 1, dims: []int{math.MaxInt / 2, 3}, wantErr: "overflow"},
		{name: "byte size overflows", size: 4, dims: []int{math.MaxInt / 2, 2}, wantErr: "overflow"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := dimBytes(tc.size, "test", tc.dims...)
			if tc.wantErr != "" {
				if !errors.Is(err, ErrInvalid) {
					t.Fatalf("want ErrInvalid, got %v", err)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("error %q does not mention %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %d bytes, want %d", got, tc.want)
			}
		})
	}
}

func TestCheckSpan(t *testing.T) {
	tests := []struct {
		name    string
		size    int
		off     int
		n       int
		wantErr string
	}{
		{name: "whole buffer", size: 16, off: 0, n: 16},
		{name: "empty range at end", size: 16, off: 16, n: 0},
		{name: "empty allocation", size: 0, off: 0, n: 0},
		{name: "tail", size: 16, off: 8, n: 8},
		{name: "offset past end", size: 16, off: 17, n: 0, wantErr: "past the end"},
		{name: "length past end", size: 16, off: 0, n: 17, wantErr: "exceed"},
		{name: "negative offset", size: 16, off: -1, n: 4, wantErr: "negative offset"},
		{name: "negative length", size: 16, off: 0, n: -4, wantErr: "negative length"},
		{name: "huge offset cannot wrap", size: math.MaxInt, off: math.MaxInt - 2, n: 8, wantErr: "exceed"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := checkSpan(tc.size, tc.off, tc.n, "test")
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("want ErrInvalid, got %v", err)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error %q does not mention %q", err, tc.wantErr)
			}
		})
	}
}

func TestUsable(t *testing.T) {
	tests := []struct {
		name    string
		buffers []*Buffer
		wantErr string
	}{
		{name: "one buffer", buffers: []*Buffer{sized(16, 0)}},
		{name: "same device", buffers: []*Buffer{sized(16, 0), sized(32, 0)}},
		{name: "nil buffer", buffers: []*Buffer{sized(16, 0), nil}, wantErr: "nil"},
		{name: "freed buffer", buffers: []*Buffer{freedSized(16)}, wantErr: "freed"},
		{name: "device mismatch", buffers: []*Buffer{sized(16, 0), sized(16, 1)}, wantErr: "device"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := usable("test", tc.buffers...)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("want ErrInvalid, got %v", err)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error %q does not mention %q", err, tc.wantErr)
			}
		})
	}
}

func TestValidators(t *testing.T) {
	tests := []struct {
		name   string
		call   func() error
		wantOK bool
		want   string
	}{
		// gemm: A[M,K] x B[K,N] -> out[M,N], 4x4x4 float32.
		{name: "gemm valid", wantOK: true, call: func() error {
			return validateGemm("gemm", sized(64, 0), sized(64, 0), sized(64, 0), 4, 4, 4, 4)
		}},
		{name: "gemm undersized lhs", want: "lhs needs 64 bytes", call: func() error {
			return validateGemm("gemm", sized(60, 0), sized(64, 0), sized(64, 0), 4, 4, 4, 4)
		}},
		{name: "gemm undersized out", want: "out needs 64 bytes", call: func() error {
			return validateGemm("gemm", sized(64, 0), sized(64, 0), sized(0, 0), 4, 4, 4, 4)
		}},
		{name: "gemm zero K", want: "non-positive dimension", call: func() error {
			return validateGemm("gemm", sized(64, 0), sized(64, 0), sized(64, 0), 4, 4, 4, 0)
		}},
		{name: "gemm overflow", want: "overflow", call: func() error {
			return validateGemm("gemm", sized(64, 0), sized(64, 0), sized(64, 0), 4, math.MaxInt/2, 2, 4)
		}},
		{name: "gemm device mismatch", want: "device", call: func() error {
			return validateGemm("gemm", sized(64, 0), sized(64, 1), sized(64, 0), 4, 4, 4, 4)
		}},

		// elemwise / scale / reduce.
		{name: "elemwise valid", wantOK: true, call: func() error {
			return validateElemwise("elemwise", sized(16, 0), sized(16, 0), sized(16, 0), 4, 4)
		}},
		{name: "elemwise zero n", want: "non-positive", call: func() error {
			return validateElemwise("elemwise", sized(16, 0), sized(16, 0), sized(16, 0), 4, 0)
		}},
		{name: "elemwise undersized b", want: "b needs 16 bytes", call: func() error {
			return validateElemwise("elemwise", sized(16, 0), sized(8, 0), sized(16, 0), 4, 4)
		}},
		{name: "elemwise freed out", want: "freed", call: func() error {
			return validateElemwise("elemwise", sized(16, 0), sized(16, 0), freedSized(16), 4, 4)
		}},
		{name: "scale valid", wantOK: true, call: func() error {
			return validateScale("scale", sized(16, 0), sized(16, 0), 4, 4)
		}},
		{name: "scale negative n", want: "non-positive", call: func() error {
			return validateScale("scale", sized(16, 0), sized(16, 0), 4, -4)
		}},
		{name: "reduce valid", wantOK: true, call: func() error {
			return validateReduce("reduce", sized(400, 0), sized(4, 0), 4, 100)
		}},
		{name: "reduce undersized out", want: "out needs 4 bytes", call: func() error {
			return validateReduce("reduce", sized(400, 0), sized(0, 0), 4, 100)
		}},
		{name: "reduce empty source", want: "non-positive", call: func() error {
			return validateReduce("reduce", sized(400, 0), sized(4, 0), 4, 0)
		}},

		// d2d copies.
		{name: "d2d valid", wantOK: true, call: func() error {
			return validateMemcpyD2D("d2d", sized(16, 0), 8, sized(16, 0), 0, 8)
		}},
		{name: "d2d zero length", wantOK: true, call: func() error {
			return validateMemcpyD2D("d2d", sized(16, 0), 16, sized(16, 0), 16, 0)
		}},
		{name: "d2d negative dst offset", want: "negative offset", call: func() error {
			return validateMemcpyD2D("d2d", sized(16, 0), -4, sized(16, 0), 0, 8)
		}},
		{name: "d2d negative length", want: "negative length", call: func() error {
			return validateMemcpyD2D("d2d", sized(16, 0), 0, sized(16, 0), 0, -1)
		}},
		{name: "d2d dst overflow", want: "exceed", call: func() error {
			return validateMemcpyD2D("d2d", sized(16, 0), 8, sized(16, 0), 0, 16)
		}},
		{name: "d2d src overflow", want: "exceed", call: func() error {
			return validateMemcpyD2D("d2d", sized(16, 0), 0, sized(16, 0), 12, 8)
		}},
		{name: "d2d overlapping same buffer", want: "overlap", call: func() error {
			b := sized(16, 0)
			return validateMemcpyD2D("d2d", b, 4, b, 0, 8)
		}},
		{name: "d2d freed source", want: "freed", call: func() error {
			return validateMemcpyD2D("d2d", sized(16, 0), 0, freedSized(16), 0, 4)
		}},

		// transformer primitives.
		{name: "rmsnorm valid", wantOK: true, call: func() error {
			return validateRMSNorm("rmsnorm", sized(32, 0), sized(16, 0), sized(32, 0), 2, 4)
		}},
		{name: "rmsnorm zero rows", want: "non-positive", call: func() error {
			return validateRMSNorm("rmsnorm", sized(32, 0), sized(16, 0), sized(32, 0), 0, 4)
		}},
		{name: "rmsnorm undersized gamma", want: "gamma needs 16 bytes", call: func() error {
			return validateRMSNorm("rmsnorm", sized(32, 0), sized(12, 0), sized(32, 0), 2, 4)
		}},
		{name: "rope batch valid", wantOK: true, call: func() error {
			return validateRoPEBatch("rope_batch", sized(64, 0), sized(64, 0), 2, 8, 4, 2)
		}},
		{name: "rope batch heads mismatch", want: "heads", call: func() error {
			return validateRoPEBatch("rope_batch", sized(64, 0), sized(64, 0), 2, 8, 3, 2)
		}},
		{name: "rope single valid", wantOK: true, call: func() error {
			return validateRoPESingle("rope_single", sized(32, 0), sized(32, 0), 3, 8, 4, 2)
		}},
		{name: "rope single negative pos", want: "negative position", call: func() error {
			return validateRoPESingle("rope_single", sized(32, 0), sized(32, 0), -1, 8, 4, 2)
		}},
		{name: "swiglu valid", wantOK: true, call: func() error {
			return validateSwiGLU("swiglu", sized(16, 0), sized(16, 0), sized(16, 0), 4)
		}},
		{name: "swiglu undersized up", want: "up needs 16 bytes", call: func() error {
			return validateSwiGLU("swiglu", sized(16, 0), sized(12, 0), sized(16, 0), 4)
		}},
		{name: "bias add valid", wantOK: true, call: func() error {
			return validateBiasAdd("bias_add", sized(64, 0), sized(16, 0), 4, 4)
		}},
		{name: "bias add undersized bias", want: "bias needs 16 bytes", call: func() error {
			return validateBiasAdd("bias_add", sized(64, 0), sized(12, 0), 4, 4)
		}},
		{name: "embed valid", wantOK: true, call: func() error {
			return validateEmbed("embed", sized(32, 0), sized(8, 0), sized(32, 0), 2, 4)
		}},
		{name: "embed unaligned table", want: "whole number", call: func() error {
			return validateEmbed("embed", sized(30, 0), sized(8, 0), sized(32, 0), 2, 4)
		}},
		{name: "embed undersized ids", want: "ids needs 8 bytes", call: func() error {
			return validateEmbed("embed", sized(32, 0), sized(4, 0), sized(32, 0), 2, 4)
		}},
		{name: "attn scores valid", wantOK: true, call: func() error {
			return validateAttnScores("attn_scores", sized(64, 0), sized(96, 0), sized(48, 0), 2, 2, 3, 8, 4)
		}},
		{name: "attn scores heads mismatch", want: "heads", call: func() error {
			return validateAttnScores("attn_scores", sized(64, 0), sized(96, 0), sized(48, 0), 2, 2, 3, 8, 3)
		}},
		{name: "attn scores undersized attn", want: "attn needs 48 bytes", call: func() error {
			return validateAttnScores("attn_scores", sized(64, 0), sized(96, 0), sized(32, 0), 2, 2, 3, 8, 4)
		}},
		{name: "attn softmax valid", wantOK: true, call: func() error {
			return validateAttnSoftmax("attn_softmax", sized(48, 0), 2, 2, 3)
		}},
		{name: "attn softmax zero kv len", want: "non-positive", call: func() error {
			return validateAttnSoftmax("attn_softmax", sized(48, 0), 2, 2, 0)
		}},
		{name: "attn weighted valid", wantOK: true, call: func() error {
			return validateAttnWeighted("attn_weighted", sized(48, 0), sized(96, 0), sized(64, 0), 2, 2, 3, 8, 4)
		}},
		{name: "attn weighted undersized v", want: "v needs 96 bytes", call: func() error {
			return validateAttnWeighted("attn_weighted", sized(48, 0), sized(64, 0), sized(64, 0), 2, 2, 3, 8, 4)
		}},

		// host<->device transfers.
		{name: "transfer exact", wantOK: true, call: func() error {
			return checkTransfer("upload", sized(16, 0), 16)
		}},
		{name: "transfer empty buffer", wantOK: true, call: func() error {
			return checkTransfer("upload", sized(0, 0), 0)
		}},
		{name: "transfer partial", want: "does not match", call: func() error {
			return checkTransfer("upload", sized(16, 0), 8)
		}},
		{name: "transfer negative", want: "negative transfer length", call: func() error {
			return checkTransfer("upload", sized(16, 0), -4)
		}},
		{name: "transfer freed", want: "freed", call: func() error {
			return checkTransfer("upload", freedSized(16), 16)
		}},

		// host-slice shapes.
		{name: "host matmul valid", wantOK: true, call: func() error {
			empty, err := checkHostMatMul("matmul", 2, 2, 2, 4, 4, 4)
			if empty {
				return errors.New("unexpected empty shape")
			}
			return err
		}},
		{name: "host matmul empty", wantOK: true, call: func() error {
			empty, err := checkHostMatMul("matmul", 0, 2, 2, 0, 0, 0)
			if err != nil {
				return err
			}
			if !empty {
				return errors.New("want empty shape")
			}
			return nil
		}},
		{name: "host matmul undersized A", want: "A has 3 elements", call: func() error {
			_, err := checkHostMatMul("matmul", 2, 2, 2, 3, 4, 4)
			return err
		}},
		{name: "host matmul undersized C", want: "C has 3 elements", call: func() error {
			_, err := checkHostMatMul("matmul", 2, 2, 2, 4, 4, 3)
			return err
		}},
		{name: "host matmul empty shape with data", want: "non-empty operands", call: func() error {
			_, err := checkHostMatMul("matmul", 0, 2, 2, 1, 0, 0)
			return err
		}},
		{name: "host matmul negative dim", want: "negative dimension", call: func() error {
			_, err := checkHostMatMul("matmul", -1, 2, 2, 0, 0, 0)
			return err
		}},
		{name: "host operands valid", wantOK: true, call: func() error {
			return checkHostOperands("add", 4, 4, 4)
		}},
		{name: "host operands empty", wantOK: true, call: func() error {
			return checkHostOperands("add", 0, 0, 0)
		}},
		{name: "host operands mismatch", want: "does not match", call: func() error {
			return checkHostOperands("add", 4, 3, 4)
		}},
		{name: "host operands data without output", want: "does not match", call: func() error {
			return checkHostOperands("add", 0, 1, 0)
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.call()
			if tc.wantOK {
				if err != nil {
					t.Fatalf("want success, got %v", err)
				}
				return
			}
			if err == nil {
				t.Fatal("want a validation error, got nil")
			}
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("want ErrInvalid, got %v", err)
			}
			if tc.want != "" && !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not mention %q", err, tc.want)
			}
		})
	}
}

// TestOperationsRejectInvalidInput exercises the public API. Every case fails
// validation, so no allocation, copy or kernel launch happens on any build.
func TestOperationsRejectInvalidInput(t *testing.T) {
	b4 := sized(16, 0)
	tests := []struct {
		name string
		call func() error
		want string
	}{
		{name: "gemm nil buffer", want: "nil", call: func() error {
			return GemmF32(nil, b4, b4, 2, 2, 2)
		}},
		{name: "gemm freed buffer", want: "freed", call: func() error {
			return GemmF32(b4, freedSized(16), b4, 2, 2, 2)
		}},
		{name: "gemm device mismatch", want: "device", call: func() error {
			return GemmF32(b4, sized(16, 1), sized(16, 0), 2, 2, 2)
		}},
		{name: "gemm negative dimension", want: "non-positive", call: func() error {
			return GemmF32(b4, b4, b4, -2, 2, 2)
		}},
		{name: "gemm overflowing dimensions", want: "overflow", call: func() error {
			return GemmF32(b4, b4, b4, math.MaxInt/2, 2, 2)
		}},
		{name: "gemm undersized operand", want: "needs 64 bytes", call: func() error {
			return GemmF32(sized(60, 0), sized(64, 0), sized(64, 0), 4, 4, 4)
		}},
		{name: "elemwise zero elements", want: "non-positive", call: func() error {
			return ElemwiseF32(OpAdd, b4, b4, b4, 0)
		}},
		{name: "scale empty source", want: "non-positive", call: func() error {
			return ScaleF32(sized(0, 0), 2, sized(0, 0), 0)
		}},
		{name: "sum empty source", want: "non-positive", call: func() error {
			return SumF32(sized(0, 0), sized(4, 0), 0)
		}},
		{name: "d2d negative offset", want: "negative offset", call: func() error {
			return MemcpyD2D(b4, -4, b4, 0, 4)
		}},
		{name: "d2d past end", want: "exceed", call: func() error {
			return MemcpyD2D(b4, 8, sized(16, 0), 0, 16)
		}},
		{name: "d2d overlap", want: "overlap", call: func() error {
			return MemcpyD2D(b4, 4, b4, 0, 8)
		}},
		{name: "d2d freed destination", want: "freed", call: func() error {
			return MemcpyD2D(freedSized(16), 0, b4, 0, 4)
		}},
		{name: "rope single negative position", want: "negative position", call: func() error {
			return RoPESingleF32(sized(32, 0), sized(32, 0), -1, 8, 4, 2)
		}},
		{name: "rope batch heads mismatch", want: "heads", call: func() error {
			return RoPEBatchF32(sized(64, 0), sized(64, 0), 2, 8, 3, 2)
		}},
		{name: "rmsnorm zero rows", want: "non-positive", call: func() error {
			return RMSNormF32(sized(32, 0), sized(8, 0), sized(32, 0), 0, 4, 1e-6)
		}},
		{name: "attn scores heads mismatch", want: "heads", call: func() error {
			return AttnScoresF32(sized(64, 0), sized(96, 0), sized(48, 0), 2, 2, 3, 8, 3, 1, true)
		}},
		{name: "attn weighted undersized v", want: "v needs 96 bytes", call: func() error {
			return AttnWeightedF32(sized(48, 0), sized(64, 0), sized(64, 0), 2, 2, 3, 8, 4)
		}},
		{name: "embed unaligned table", want: "whole number", call: func() error {
			return EmbedF32(sized(30, 0), sized(8, 0), sized(32, 0), 2, 4)
		}},
		{name: "swiglu zero elements", want: "non-positive", call: func() error {
			return SwiGLUF32(b4, b4, b4, 0)
		}},
		{name: "upload oversized slice", want: "does not match", call: func() error {
			return sized(4, 0).Upload([]float32{1, 2})
		}},
		{name: "upload bytes into float buffer", want: "does not match", call: func() error {
			return sized(16, 0).UploadBytes(make([]byte, 4))
		}},
		{name: "download undersized slice", want: "does not match", call: func() error {
			return sized(16, 0).Download(make([]float32, 2))
		}},
		{name: "upload into freed buffer", want: "freed", call: func() error {
			return freedSized(16).Upload(make([]float32, 4))
		}},
		{name: "upload into nil buffer", want: "nil", call: func() error {
			var b *Buffer
			return b.Upload(nil)
		}},
		{name: "alloc negative f32", want: "negative element count", call: func() error {
			_, err := AllocF32(-1)
			return err
		}},
		{name: "alloc negative f64", want: "negative element count", call: func() error {
			_, err := AllocF64(-1)
			return err
		}},
		{name: "alloc negative bytes", want: "negative element count", call: func() error {
			_, err := AllocBytes(-4)
			return err
		}},
		{name: "alloc overflow", want: "overflow", call: func() error {
			_, err := AllocF32(math.MaxInt)
			return err
		}},
		{name: "matmul undersized A", want: "A has 3 elements", call: func() error {
			return MatMulFloat32(2, 2, 2, make([]float32, 3), make([]float32, 4), make([]float32, 4))
		}},
		{name: "matmul negative dimension", want: "negative dimension", call: func() error {
			return MatMulFloat32(-1, 2, 2, nil, nil, nil)
		}},
		{name: "matmul empty shape with data", want: "non-empty operands", call: func() error {
			return MatMulFloat32(0, 2, 2, []float32{1}, nil, nil)
		}},
		{name: "add mismatched operands", want: "does not match", call: func() error {
			return AddFloat32([]float32{1, 2, 3}, []float32{1}, make([]float32, 2))
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.call()
			if err == nil {
				t.Fatal("want a validation error, got nil")
			}
			if !errors.Is(err, ErrInvalid) {
				t.Fatalf("want ErrInvalid (rejected before dispatch), got %v", err)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not mention %q", err, tc.want)
			}
			if errors.Is(err, ErrUnavailable) {
				t.Fatalf("invalid input reached the backend: %v", err)
			}
		})
	}
}

// TestEmptyInputIsNoOp pins the documented no-op results for empty input.
func TestEmptyInputIsNoOp(t *testing.T) {
	empty := sized(0, 0)
	if err := empty.Upload(nil); err != nil {
		t.Fatalf("Upload(nil) on an empty buffer: %v", err)
	}
	if err := empty.UploadF64(nil); err != nil {
		t.Fatalf("UploadF64(nil) on an empty buffer: %v", err)
	}
	if err := empty.UploadBytes(nil); err != nil {
		t.Fatalf("UploadBytes(nil) on an empty buffer: %v", err)
	}
	if err := empty.Download(nil); err != nil {
		t.Fatalf("Download(nil) on an empty buffer: %v", err)
	}
	if err := empty.DownloadF64(nil); err != nil {
		t.Fatalf("DownloadF64(nil) on an empty buffer: %v", err)
	}
	if err := empty.Zero(); err != nil {
		t.Fatalf("Zero on an empty buffer: %v", err)
	}
	if err := MemcpyD2D(empty, 0, empty, 0, 0); err != nil {
		t.Fatalf("zero-length d2d copy: %v", err)
	}
	if err := MatMulFloat32(0, 0, 0, nil, nil, nil); err != nil {
		t.Fatalf("empty matmul: %v", err)
	}
	if err := AddFloat32(nil, nil, nil); err != nil {
		t.Fatalf("empty add: %v", err)
	}
	if err := ScaleFloat32(nil, 2, nil); err != nil {
		t.Fatalf("empty scale: %v", err)
	}
	if v, err := SumFloat32(nil); err != nil || v != 0 {
		t.Fatalf("empty sum: got %v, %v", v, err)
	}
	if v, err := MaxFloat64(nil); err != nil || v != 0 {
		t.Fatalf("empty max: got %v, %v", v, err)
	}

	// AllocF32(0) is a legal empty allocation whose operations are no-ops.
	alloc, err := AllocF32(0)
	if err != nil {
		t.Fatalf("AllocF32(0): %v", err)
	}
	if alloc.Bytes() != 0 {
		t.Fatalf("AllocF32(0) reports %d bytes", alloc.Bytes())
	}
	if err := alloc.Upload(nil); err != nil {
		t.Fatalf("Upload(nil) on AllocF32(0): %v", err)
	}
	alloc.Free()

	// Nil-buffer methods are safe.
	var nilBuf *Buffer
	nilBuf.Free()
	if got := nilBuf.Bytes(); got != 0 {
		t.Fatalf("nil Bytes() = %d, want 0", got)
	}
	if got := nilBuf.DevicePointer(); got != nil {
		t.Fatalf("nil DevicePointer() = %v, want nil", got)
	}
}
