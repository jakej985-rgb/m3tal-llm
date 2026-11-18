//go:build !cuda || !cgo || !amd64

package gpu

// This stub keeps the package API available on builds without CUDA support
// (the default) and on non-amd64 architectures. Arguments are validated with
// the same rules as the real CUDA implementation (see bounds.go), so calling
// code sees identical input errors either way; a valid request reports
// ErrUnavailable so callers can fall back to the CPU path.

import (
	"errors"
	"fmt"
	"math"
	"unsafe"
)

// ErrUnavailable is returned when CUDA support is not compiled in or no
// usable device exists.
var ErrUnavailable = errors.New("CUDA backend unavailable: rebuild with CGO_ENABLED=1 go build -tags cuda (see docs/CUDA.md)")

// Buffer is an opaque device allocation placeholder for stub builds.
type Buffer struct {
	bytes  int
	device int
	freed  bool
}

// currentDevice reports the device new allocations land on. Stub builds have
// no device; buffers record device 0 so mixed-device validation still runs.
func currentDevice() int { return 0 }

func alloc(bytes int, what string) (*Buffer, error) {
	if bytes < 0 {
		return nil, invalidf("%s: negative size %d", what, bytes)
	}
	if bytes == 0 {
		return &Buffer{}, nil
	}
	return nil, fmt.Errorf("%w: %s allocation of %d bytes", ErrUnavailable, what, bytes)
}

// AllocF32 allocates n float32 elements on the device.
func AllocF32(n int) (*Buffer, error) {
	bytes, err := elemBytes(n, 4, "float32 allocation")
	if err != nil {
		return nil, err
	}
	return alloc(bytes, "float32")
}

// AllocF64 allocates n float64 elements on the device.
func AllocF64(n int) (*Buffer, error) {
	bytes, err := elemBytes(n, 8, "float64 allocation")
	if err != nil {
		return nil, err
	}
	return alloc(bytes, "float64")
}

// AllocBytes allocates raw bytes on the device.
func AllocBytes(n int) (*Buffer, error) {
	bytes, err := elemBytes(n, 1, "byte allocation")
	if err != nil {
		return nil, err
	}
	return alloc(bytes, "bytes")
}

// Free releases the device allocation. It is idempotent and safe on nil.
func (b *Buffer) Free() {
	if b != nil {
		b.freed = true
	}
}

// Bytes returns the allocation size in bytes (0 for a nil buffer).
func (b *Buffer) Bytes() int {
	if b == nil {
		return 0
	}
	return b.bytes
}

// Upload copies a host float32 slice into the buffer.
func (b *Buffer) Upload(src []float32) error {
	n, err := elemBytes(len(src), 4, "upload")
	if err != nil {
		return err
	}
	return b.transfer("upload float32", n)
}

// UploadF64 copies a host float64 slice into the buffer.
func (b *Buffer) UploadF64(src []float64) error {
	n, err := elemBytes(len(src), 8, "upload")
	if err != nil {
		return err
	}
	return b.transfer("upload float64", n)
}

// UploadBytes copies raw host bytes into the buffer.
func (b *Buffer) UploadBytes(src []byte) error {
	n, err := elemBytes(len(src), 1, "upload")
	if err != nil {
		return err
	}
	return b.transfer("upload bytes", n)
}

// Download copies the buffer contents into a host float32 slice.
func (b *Buffer) Download(dst []float32) error {
	n, err := elemBytes(len(dst), 4, "download")
	if err != nil {
		return err
	}
	return b.transfer("download float32", n)
}

// DownloadF64 copies the buffer contents into a host float64 slice.
func (b *Buffer) DownloadF64(dst []float64) error {
	n, err := elemBytes(len(dst), 8, "download")
	if err != nil {
		return err
	}
	return b.transfer("download float64", n)
}

// transfer validates a host<->device copy of n bytes; a zero-byte copy is a
// documented no-op on an empty buffer even without a device.
func (b *Buffer) transfer(what string, n int) error {
	if err := checkTransfer(what, b, n); err != nil {
		return err
	}
	if n == 0 {
		return nil
	}
	return unavail(what)
}

// Zero fills the buffer with zero bytes.
func (b *Buffer) Zero() error {
	if err := usable("zero", b); err != nil {
		return err
	}
	if b.bytes == 0 {
		return nil
	}
	return unavail("zero")
}

// DevicePointer exposes the raw device pointer (always nil on stubs).
func (b *Buffer) DevicePointer() unsafe.Pointer { return nil }

// DeviceInfo describes the primary CUDA device (stub builds never have one).
type DeviceInfo struct {
	Count      int
	Name       string
	Major      int
	Minor      int
	DeviceArch string
}

// ProbeDevice always fails on stub builds.
func ProbeDevice() (DeviceInfo, error) { return DeviceInfo{}, ErrUnavailable }

// LastError returns the most recent CUDA error, or nil.
func LastError() error { return nil }

// Sync blocks until all queued work on the device completes.
func Sync() error { return ErrUnavailable }

// DecodeMaxKeyF32 converts an order-preserving uint32 key to float32.
func DecodeMaxKeyF32(key uint32) float32 {
	var u uint32
	if key&0x80000000 != 0 {
		u = key & 0x7fffffff
	} else {
		u = ^key
	}
	return math.Float32frombits(u)
}

func unavail(what string) error {
	return fmt.Errorf("%w: %s", ErrUnavailable, what)
}

// MemcpyD2D copies bytes between device buffers (stub).
func MemcpyD2D(dst *Buffer, dstOff int, src *Buffer, srcOff int, n int) error {
	if err := validateMemcpyD2D("d2d memcpy", dst, dstOff, src, srcOff, n); err != nil {
		return err
	}
	if n == 0 {
		return nil
	}
	return unavail("d2d memcpy")
}

// GemmF32 computes C[M,N] = A[M,K] * B[K,N] on device buffers.
func GemmF32(A, B, C *Buffer, M, N, K int) error {
	if err := validateGemm("gemm_f32", A, B, C, 4, M, N, K); err != nil {
		return err
	}
	return unavail("gemm_f32")
}

// GemmF64 computes C[M,N] = A[M,K] * B[K,N] on device buffers.
func GemmF64(A, B, C *Buffer, M, N, K int) error {
	if err := validateGemm("gemm_f64", A, B, C, 8, M, N, K); err != nil {
		return err
	}
	return unavail("gemm_f64")
}

// ElemwiseF32 computes out[i] = a[i] OP b[i] on device buffers.
func ElemwiseF32(op int, A, B, Out *Buffer, n int) error {
	if err := validateElemwise("elemwise_f32", A, B, Out, 4, n); err != nil {
		return err
	}
	return unavail("elemwise_f32")
}

// ElemwiseF64 computes out[i] = a[i] OP b[i] on device buffers.
func ElemwiseF64(op int, A, B, Out *Buffer, n int) error {
	if err := validateElemwise("elemwise_f64", A, B, Out, 8, n); err != nil {
		return err
	}
	return unavail("elemwise_f64")
}

// ScaleF32 computes out[i] = src[i] * alpha on device buffers.
func ScaleF32(Src *Buffer, alpha float32, Out *Buffer, n int) error {
	if err := validateScale("scale_f32", Src, Out, 4, n); err != nil {
		return err
	}
	return unavail("scale_f32")
}

// ScaleF64 computes out[i] = src[i] * alpha on device buffers.
func ScaleF64(Src *Buffer, alpha float64, Out *Buffer, n int) error {
	if err := validateScale("scale_f64", Src, Out, 8, n); err != nil {
		return err
	}
	return unavail("scale_f64")
}

// SumF32 reduces a device float32 buffer to a single value.
func SumF32(Src, Out *Buffer, n int) error {
	if err := validateReduce("sum_f32", Src, Out, 4, n); err != nil {
		return err
	}
	return unavail("sum_f32")
}

// MaxF32 reduces a device float32 buffer to its maximum.
func MaxF32(Src, Out *Buffer, n int) error {
	if err := validateReduce("max_f32", Src, Out, 4, n); err != nil {
		return err
	}
	return unavail("max_f32")
}

// SumF64 reduces a device float64 buffer to a single value.
func SumF64(Src, Out *Buffer, n int) error {
	if err := validateReduce("sum_f64", Src, Out, 8, n); err != nil {
		return err
	}
	return unavail("sum_f64")
}

// MaxF64 reduces a device float64 buffer to its maximum.
func MaxF64(Src, Out *Buffer, n int) error {
	if err := validateReduce("max_f64", Src, Out, 8, n); err != nil {
		return err
	}
	return unavail("max_f64")
}

// MatMulFloat32 computes C[M,N] = A[M,K] * B[K,N] from host slices.
func MatMulFloat32(M, N, K int, A, B, C []float32) error {
	empty, err := checkHostMatMul("matmul_f32", M, N, K, len(A), len(B), len(C))
	if err != nil {
		return err
	}
	if empty {
		return nil
	}
	return unavail("matmul_f32")
}

// MatMulFloat64 computes C[M,N] = A[M,K] * B[K,N] from host slices.
func MatMulFloat64(M, N, K int, A, B, C []float64) error {
	empty, err := checkHostMatMul("matmul_f64", M, N, K, len(A), len(B), len(C))
	if err != nil {
		return err
	}
	if empty {
		return nil
	}
	return unavail("matmul_f64")
}

// AddFloat32 computes out=a+b on host slices.
func AddFloat32(a, b, out []float32) error { return hostElemwise("add_f32", a, b, out) }

// SubFloat32 computes out=a-b on host slices.
func SubFloat32(a, b, out []float32) error { return hostElemwise("sub_f32", a, b, out) }

// MulFloat32 computes out=a*b on host slices.
func MulFloat32(a, b, out []float32) error { return hostElemwise("mul_f32", a, b, out) }

// DivFloat32 computes out=a/b on host slices.
func DivFloat32(a, b, out []float32) error { return hostElemwise("div_f32", a, b, out) }

// AddFloat64 computes out=a+b on host slices.
func AddFloat64(a, b, out []float64) error { return hostElemwise64("add_f64", a, b, out) }

// SubFloat64 computes out=a-b on host slices.
func SubFloat64(a, b, out []float64) error { return hostElemwise64("sub_f64", a, b, out) }

// MulFloat64 computes out=a*b on host slices.
func MulFloat64(a, b, out []float64) error { return hostElemwise64("mul_f64", a, b, out) }

// DivFloat64 computes out=a/b on host slices.
func DivFloat64(a, b, out []float64) error { return hostElemwise64("div_f64", a, b, out) }

func hostElemwise(what string, a, b, out []float32) error {
	if err := checkHostOperands(what, len(out), len(a), len(b)); err != nil {
		return err
	}
	if len(out) == 0 {
		return nil
	}
	return unavail(what)
}

func hostElemwise64(what string, a, b, out []float64) error {
	if err := checkHostOperands(what, len(out), len(a), len(b)); err != nil {
		return err
	}
	if len(out) == 0 {
		return nil
	}
	return unavail(what)
}

// ScaleFloat32 computes out[i] = src[i] * alpha on host slices.
func ScaleFloat32(src []float32, alpha float32, out []float32) error {
	if err := checkHostOperands("scale_f32", len(out), len(src)); err != nil {
		return err
	}
	if len(out) == 0 {
		return nil
	}
	return unavail("scale_f32")
}

// ScaleFloat64 computes out[i] = src[i] * alpha on host slices.
func ScaleFloat64(src []float64, alpha float64, out []float64) error {
	if err := checkHostOperands("scale_f64", len(out), len(src)); err != nil {
		return err
	}
	if len(out) == 0 {
		return nil
	}
	return unavail("scale_f64")
}

// SumFloat32 reduces a host float32 slice on the GPU.
func SumFloat32(src []float32) (float32, error) {
	if len(src) == 0 {
		return 0, nil
	}
	return 0, unavail("sum_f32")
}

// MaxFloat32 reduces a host float32 slice on the GPU.
func MaxFloat32(src []float32) (float32, error) {
	if len(src) == 0 {
		return 0, nil
	}
	return 0, unavail("max_f32")
}

// SumFloat64 reduces a host float64 slice on the GPU.
func SumFloat64(src []float64) (float64, error) {
	if len(src) == 0 {
		return 0, nil
	}
	return 0, unavail("sum_f64")
}

// MaxFloat64 reduces a host float64 slice on the GPU.
func MaxFloat64(src []float64) (float64, error) {
	if len(src) == 0 {
		return 0, nil
	}
	return 0, unavail("max_f64")
}

// RMSNormF32 stub.
func RMSNormF32(x, gamma, out *Buffer, rows, width int, eps float32) error {
	if err := validateRMSNorm("rmsnorm_f32", x, gamma, out, rows, width); err != nil {
		return err
	}
	return unavail("rmsnorm_f32")
}

// RoPEBatchF32 stub.
func RoPEBatchF32(q, k *Buffer, rows, width, headDim, heads int) error {
	if err := validateRoPEBatch("rope_batch", q, k, rows, width, headDim, heads); err != nil {
		return err
	}
	return unavail("rope_batch")
}

// RoPESingleF32 stub.
func RoPESingleF32(q, k *Buffer, pos, width, headDim, heads int) error {
	if err := validateRoPESingle("rope_single", q, k, pos, width, headDim, heads); err != nil {
		return err
	}
	return unavail("rope_single")
}

// SwiGLUF32 stub.
func SwiGLUF32(gate, up, out *Buffer, n int) error {
	if err := validateSwiGLU("swiglu", gate, up, out, n); err != nil {
		return err
	}
	return unavail("swiglu")
}

// BiasAddF32 stub.
func BiasAddF32(c, bias *Buffer, M, N int) error {
	if err := validateBiasAdd("bias_add", c, bias, M, N); err != nil {
		return err
	}
	return unavail("bias_add")
}

// EmbedF32 stub.
func EmbedF32(emb, ids, out *Buffer, rows, width int) error {
	if err := validateEmbed("embed", emb, ids, out, rows, width); err != nil {
		return err
	}
	return unavail("embed")
}

// AttnScoresF32 stub.
func AttnScoresF32(q, k, attn *Buffer, heads, qRows, kvLen, width, headDim int,
	scale float32, causal bool) error {
	if err := validateAttnScores("attn_scores", q, k, attn, heads, qRows, kvLen, width, headDim); err != nil {
		return err
	}
	return unavail("attn_scores")
}

// AttnSoftmaxF32 stub.
func AttnSoftmaxF32(attn *Buffer, heads, qRows, kvLen int) error {
	if err := validateAttnSoftmax("attn_softmax", attn, heads, qRows, kvLen); err != nil {
		return err
	}
	return unavail("attn_softmax")
}

// AttnWeightedF32 stub.
func AttnWeightedF32(attn, v, out *Buffer, heads, qRows, kvLen, width, headDim int) error {
	if err := validateAttnWeighted("attn_weighted", attn, v, out, heads, qRows, kvLen, width, headDim); err != nil {
		return err
	}
	return unavail("attn_weighted")
}
