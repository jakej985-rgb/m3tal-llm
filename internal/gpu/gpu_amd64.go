//go:build cuda && cgo && amd64

// Package gpu provides CUDA device buffers and kernel launches for
// m3tal-llm tensor primitives. It is only compiled with `-tags cuda` and
// cgo on amd64; other builds get the stub in gpu_stub.go.
//
// Build prerequisite: run `go generate ./internal/gpu` once with nvcc in
// PATH to produce kernels_amd64.o (see docs/CUDA.md).
package gpu

/*
#cgo LDFLAGS: -lcudart ${SRCDIR}/kernels_amd64.o
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

// Device memory management (CUDA runtime API).
static void* m3tal_gpu_malloc(size_t bytes) {
	void* p = NULL;
	if (bytes == 0) return NULL;
	if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
	return p;
}
static void m3tal_gpu_free(void* p) { if (p) cudaFree(p); }
static int m3tal_gpu_memcpy_h2d(void* d, const void* h, size_t n) {
	return (int)cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
}
static int m3tal_gpu_memcpy_d2h(void* h, const void* d, size_t n) {
	return (int)cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost);
}
static int m3tal_gpu_memset(void* d, int v, size_t n) {
	return (int)cudaMemset(d, v, n);
}
static int m3tal_gpu_sync(void) { return (int)cudaDeviceSynchronize(); }
static int m3tal_gpu_last_error(void) { return (int)cudaGetLastError(); }

// Device enumeration for backend probing.
static int m3tal_cuda_device_count() {
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess) return -1;
	return count;
}
static void m3tal_cuda_device_info(int device, char *name, int nameLen, int *major, int *minor) {
	struct cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return;
	strncpy(name, prop.name, (size_t)nameLen - 1);
	name[nameLen - 1] = '\0';
	*major = prop.major;
	*minor = prop.minor;
}
// Device new allocations land on; -1 when the runtime cannot be queried.
static int m3tal_cuda_current_device(void) {
	int device = 0;
	if (cudaGetDevice(&device) != cudaSuccess) return -1;
	return device;
}

// Kernel launchers implemented in kernels.cu (extern "C").
int m3tal_launch_gemm_f32(const void* A, const void* B, void* C, int M, int N, int K);
int m3tal_launch_gemm_f64(const void* A, const void* B, void* C, int M, int N, int K);
int m3tal_launch_elemwise_f32(int op, const void* a, const void* b, void* out, long long n);
int m3tal_launch_elemwise_f64(int op, const void* a, const void* b, void* out, long long n);
int m3tal_launch_scale_f32(const void* src, float alpha, void* out, long long n);
int m3tal_launch_scale_f64(const void* src, double alpha, void* out, long long n);
int m3tal_launch_sum_f32(const void* src, void* out, long long n);
int m3tal_launch_max_f32(const void* src, void* outKey, long long n);
int m3tal_launch_sum_f64(const void* src, void* out, long long n);
int m3tal_launch_max_f64(const void* src, void* out, long long n);

// Transformer/lm kernels.
int m3tal_launch_rmsnorm_f32(const void* x, const void* gamma, void* out,
                             int rows, int width, float eps);
int m3tal_launch_rope_batch_f32(void* q, void* k, int rows, int width,
                                int headDim, int heads);
int m3tal_launch_rope_single_f32(void* q, void* k, int pos, int width,
                                 int headDim, int heads);
int m3tal_launch_swiglu_f32(const void* gate, const void* up, void* out,
                            long long n);
int m3tal_launch_bias_add_f32(void* c, const void* bias, int M, int N);
int m3tal_launch_embed_f32(const void* emb, const void* ids, void* out,
                           int rows, int width);
int m3tal_launch_attn_scores_f32(const void* q, const void* k, void* attn,
                                 int heads, int qRows, int kvLen, int width,
                                 int headDim, float scale, int causal);
int m3tal_launch_attn_softmax_f32(void* attn, int heads, int qRows, int kvLen);
int m3tal_launch_attn_weighted_f32(const void* attn, const void* v,
                                   void* out, int heads, int qRows, int kvLen,
                                   int width, int headDim);

static int m3tal_gpu_memcpy_d2d(void* d, const void* s, size_t n) {
	return (int)cudaMemcpy(d, s, n, cudaMemcpyDeviceToDevice);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"unsafe"
)

// ErrUnavailable is returned when CUDA support is not compiled in or no
// usable device exists.
var ErrUnavailable = errors.New("CUDA backend unavailable")

var lastErr atomic.Value // stores error

// DeviceInfo describes the primary CUDA device visible to the runtime.
type DeviceInfo struct {
	Count      int
	Name       string
	Major      int
	Minor      int
	DeviceArch string
}

// ProbeDevice enumerates CUDA devices through the runtime API. It reports
// what the runtime can actually use, not just what the driver exposes.
func ProbeDevice() (DeviceInfo, error) {
	count := int(C.m3tal_cuda_device_count())
	if count < 0 {
		return DeviceInfo{}, fmt.Errorf("%w: CUDA runtime cannot enumerate devices (check driver vs runtime mismatch; see docs/CUDA.md)", ErrUnavailable)
	}
	if count == 0 {
		return DeviceInfo{}, fmt.Errorf("%w: no CUDA-capable NVIDIA devices detected", ErrUnavailable)
	}
	nameBuf := make([]byte, 256)
	var major, minor C.int
	C.m3tal_cuda_device_info(0, (*C.char)(unsafe.Pointer(&nameBuf[0])), C.int(len(nameBuf)), &major, &minor)
	name := C.GoString((*C.char)(unsafe.Pointer(&nameBuf[0])))
	return DeviceInfo{
		Count:      count,
		Name:       name,
		Major:      int(major),
		Minor:      int(minor),
		DeviceArch: fmt.Sprintf("sm_%d%d", int(major), int(minor)),
	}, nil
}

// LastError returns the most recent CUDA error observed by this package,
// or nil. Kernel dispatch falls back to CPU on error; callers that need
// diagnostics can inspect this. Inputs rejected by boundary validation never
// reach the device, so they do not set it.
func LastError() error {
	if v := lastErr.Load(); v != nil {
		return v.(error)
	}
	return nil
}

func recordErr(err error) {
	if err != nil {
		lastErr.Store(err)
		C.m3tal_gpu_sync() // drain the device so the context is usable again
	}
}

func cudaErr(code C.int, what string) error {
	if code == 0 {
		return nil
	}
	return fmt.Errorf("cuda %s failed: error %d", what, int(code))
}

// Buffer is a device-resident allocation. Ownership is explicit: callers
// must Free it (defer works). Buffers are not safe for concurrent use.
//
// Every operation validates the buffer (nil, freed, size, device) before it
// touches device memory; see bounds.go.
type Buffer struct {
	ptr    unsafe.Pointer
	bytes  int
	device int
	freed  bool
}

// currentDevice reports the device new allocations land on.
func currentDevice() int {
	if d := int(C.m3tal_cuda_current_device()); d >= 0 {
		return d
	}
	return 0
}

// AllocF32 allocates n float32 elements on the device. n == 0 returns an
// empty buffer whose operations are documented no-ops; n < 0 and sizes that
// would overflow the byte count are rejected.
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

func alloc(bytes int, what string) (*Buffer, error) {
	if bytes < 0 {
		return nil, invalidf("%s: negative size %d", what, bytes)
	}
	device := currentDevice()
	if bytes == 0 {
		return &Buffer{device: device}, nil
	}
	p := C.m3tal_gpu_malloc(C.size_t(bytes))
	if p == nil {
		err := fmt.Errorf("%w: cudaMalloc of %d bytes (%s) failed", ErrUnavailable, bytes, what)
		recordErr(err)
		return nil, err
	}
	return &Buffer{ptr: p, bytes: bytes, device: device}, nil
}

// Free releases the device allocation. It is idempotent and safe on nil.
func (b *Buffer) Free() {
	if b == nil || b.freed {
		return
	}
	C.m3tal_gpu_free(b.ptr)
	b.ptr = nil
	b.freed = true
}

// Bytes returns the allocation size in bytes (0 for a nil buffer).
func (b *Buffer) Bytes() int {
	if b == nil {
		return 0
	}
	return b.bytes
}

// Upload copies a host float32 slice into the buffer. The slice must fill the
// buffer exactly; an empty slice is a no-op only for an empty buffer.
func (b *Buffer) Upload(src []float32) error {
	n, err := elemBytes(len(src), 4, "upload")
	if err != nil {
		return err
	}
	return b.transfer("upload", unsafe.Pointer(sliceData(src)), n, true, "float32")
}

// UploadF64 copies a host float64 slice into the buffer.
func (b *Buffer) UploadF64(src []float64) error {
	n, err := elemBytes(len(src), 8, "upload")
	if err != nil {
		return err
	}
	return b.transfer("upload", unsafe.Pointer(sliceData(src)), n, true, "float64")
}

// UploadBytes copies raw host bytes into the buffer (e.g. int32 ids).
func (b *Buffer) UploadBytes(src []byte) error {
	n, err := elemBytes(len(src), 1, "upload")
	if err != nil {
		return err
	}
	return b.transfer("upload", unsafe.Pointer(sliceData(src)), n, true, "bytes")
}

// Download copies the buffer contents into a host float32 slice. The slice
// must match the buffer size exactly.
func (b *Buffer) Download(dst []float32) error {
	n, err := elemBytes(len(dst), 4, "download")
	if err != nil {
		return err
	}
	return b.transfer("download", unsafe.Pointer(sliceData(dst)), n, false, "float32")
}

// DownloadF64 copies the buffer contents into a host float64 slice.
func (b *Buffer) DownloadF64(dst []float64) error {
	n, err := elemBytes(len(dst), 8, "download")
	if err != nil {
		return err
	}
	return b.transfer("download", unsafe.Pointer(sliceData(dst)), n, false, "float64")
}

// sliceData returns the first element address of s, or nil when empty, so an
// empty slice never produces an out-of-range index.
func sliceData[T any](s []T) *T {
	if len(s) == 0 {
		return nil
	}
	return &s[0]
}

// transfer validates and performs a host<->device copy of n bytes.
func (b *Buffer) transfer(what string, h unsafe.Pointer, n int, upload bool, element string) error {
	if err := checkTransfer(what+" "+element, b, n); err != nil {
		return err
	}
	if n == 0 {
		return nil
	}
	var code C.int
	if upload {
		code = C.m3tal_gpu_memcpy_h2d(b.ptr, h, C.size_t(n))
	} else {
		code = C.m3tal_gpu_memcpy_d2h(h, b.ptr, C.size_t(n))
	}
	dir := "D2H"
	if upload {
		dir = "H2D"
	}
	if err := cudaErr(code, dir+" memcpy"); err != nil {
		recordErr(err)
		return err
	}
	return nil
}

// Zero fills the buffer with zero bytes. It is a no-op on an empty buffer.
func (b *Buffer) Zero() error {
	if err := usable("zero", b); err != nil {
		return err
	}
	if b.bytes == 0 {
		return nil
	}
	if err := cudaErr(C.m3tal_gpu_memset(b.ptr, 0, C.size_t(b.bytes)), "memset"); err != nil {
		recordErr(err)
		return err
	}
	return nil
}

// Sync blocks until all queued work on the device completes.
func Sync() error {
	return cudaErr(C.m3tal_gpu_sync(), "device synchronize")
}

// DevicePointer exposes the raw device pointer for advanced callers that
// need pointer arithmetic (e.g. offset copies). It bypasses the bounds
// checking in this package, so prefer the typed helpers; a nil result means
// the buffer is freed, empty or nil.
func (b *Buffer) DevicePointer() unsafe.Pointer {
	if b == nil {
		return nil
	}
	return b.ptr
}

// MemcpyD2D copies n bytes between two device buffers; dstOff/srcOff are
// byte offsets into each buffer (used for appending to growable caches).
// Offsets, lengths, buffer state and overlap are validated first.
func MemcpyD2D(dst *Buffer, dstOff int, src *Buffer, srcOff int, n int) error {
	if err := validateMemcpyD2D("d2d memcpy", dst, dstOff, src, srcOff, n); err != nil {
		return err
	}
	if n == 0 {
		return nil
	}
	d := unsafe.Pointer(uintptr(dst.ptr) + uintptr(dstOff))
	s := unsafe.Pointer(uintptr(src.ptr) + uintptr(srcOff))
	if err := cudaErr(C.m3tal_gpu_memcpy_d2d(d, s, C.size_t(n)), "D2D memcpy"); err != nil {
		recordErr(err)
		return err
	}
	return nil
}

// launch checks the launcher's return code and records failures.
func launch(code C.int, what string) error {
	if err := cudaErr(code, what); err != nil {
		recordErr(err)
		return err
	}
	return nil
}

// ---------------------------------------------------------------------------
// Device-resident operations (buffers already on the GPU)
//
// Each validates its buffers and shape before launching, so an invalid
// request errors instead of touching device memory.
// ---------------------------------------------------------------------------

// GemmF32 computes Out[M,N] = A[M,K] * B[K,N] on device buffers.
func GemmF32(A, B, Out *Buffer, M, N, K int) error {
	if err := validateGemm("gemm_f32", A, B, Out, 4, M, N, K); err != nil {
		return err
	}
	return launch(C.m3tal_launch_gemm_f32(A.ptr, B.ptr, Out.ptr, C.int(M), C.int(N), C.int(K)), "gemm_f32 launch")
}

// GemmF64 computes Out[M,N] = A[M,K] * B[K,N] on device buffers.
func GemmF64(A, B, Out *Buffer, M, N, K int) error {
	if err := validateGemm("gemm_f64", A, B, Out, 8, M, N, K); err != nil {
		return err
	}
	return launch(C.m3tal_launch_gemm_f64(A.ptr, B.ptr, Out.ptr, C.int(M), C.int(N), C.int(K)), "gemm_f64 launch")
}

// ElemwiseF32 computes out[i] = a[i] OP b[i] on device buffers.
func ElemwiseF32(op int, A, B, Out *Buffer, n int) error {
	if err := validateElemwise("elemwise_f32", A, B, Out, 4, n); err != nil {
		return err
	}
	return launch(C.m3tal_launch_elemwise_f32(C.int(op), A.ptr, B.ptr, Out.ptr, C.longlong(n)), "elemwise_f32 launch")
}

// ElemwiseF64 computes out[i] = a[i] OP b[i] on device buffers.
func ElemwiseF64(op int, A, B, Out *Buffer, n int) error {
	if err := validateElemwise("elemwise_f64", A, B, Out, 8, n); err != nil {
		return err
	}
	return launch(C.m3tal_launch_elemwise_f64(C.int(op), A.ptr, B.ptr, Out.ptr, C.longlong(n)), "elemwise_f64 launch")
}

// ScaleF32 computes out[i] = src[i] * alpha on device buffers.
func ScaleF32(Src *Buffer, alpha float32, Out *Buffer, n int) error {
	if err := validateScale("scale_f32", Src, Out, 4, n); err != nil {
		return err
	}
	return launch(C.m3tal_launch_scale_f32(Src.ptr, C.float(alpha), Out.ptr, C.longlong(n)), "scale_f32 launch")
}

// ScaleF64 computes out[i] = src[i] * alpha on device buffers.
func ScaleF64(Src *Buffer, alpha float64, Out *Buffer, n int) error {
	if err := validateScale("scale_f64", Src, Out, 8, n); err != nil {
		return err
	}
	return launch(C.m3tal_launch_scale_f64(Src.ptr, C.double(alpha), Out.ptr, C.longlong(n)), "scale_f64 launch")
}

// SumF32 reduces a device float32 buffer to a single value.
// The result is written into Out (4 bytes) and must be downloaded by the caller.
func SumF32(Src, Out *Buffer, n int) error {
	if err := validateReduce("sum_f32", Src, Out, 4, n); err != nil {
		return err
	}
	if err := Out.Zero(); err != nil {
		return err
	}
	return launch(C.m3tal_launch_sum_f32(Src.ptr, Out.ptr, C.longlong(n)), "sum_f32 launch")
}

// MaxF32 reduces a device float32 buffer to its maximum.
// Out holds an order-preserving uint32 key; decode with DecodeMaxKeyF32
// after downloading the raw bits.
func MaxF32(Src, Out *Buffer, n int) error {
	if err := validateReduce("max_f32", Src, Out, 4, n); err != nil {
		return err
	}
	if err := Out.Zero(); err != nil {
		return err
	}
	return launch(C.m3tal_launch_max_f32(Src.ptr, Out.ptr, C.longlong(n)), "max_f32 launch")
}

// DecodeMaxKeyF32 converts the order-preserving uint32 key produced by
// MaxF32 back to a float32.
func DecodeMaxKeyF32(key uint32) float32 {
	var u uint32
	if key&0x80000000 != 0 {
		u = key & 0x7fffffff
	} else {
		u = ^key
	}
	return math.Float32frombits(u)
}

// SumF64 reduces a device float64 buffer to a single value.
func SumF64(Src, Out *Buffer, n int) error {
	if err := validateReduce("sum_f64", Src, Out, 8, n); err != nil {
		return err
	}
	return launch(C.m3tal_launch_sum_f64(Src.ptr, Out.ptr, C.longlong(n)), "sum_f64 launch")
}

// MaxF64 reduces a device float64 buffer to its maximum.
func MaxF64(Src, Out *Buffer, n int) error {
	if err := validateReduce("max_f64", Src, Out, 8, n); err != nil {
		return err
	}
	return launch(C.m3tal_launch_max_f64(Src.ptr, Out.ptr, C.longlong(n)), "max_f64 launch")
}

// ---------------------------------------------------------------------------
// Transparent host-slice operations (H2D -> kernel -> D2H per call).
// Used by the kernel dispatch layer; higher-level code should prefer
// device-resident buffers to avoid per-call transfer overhead.
// ---------------------------------------------------------------------------

// MatMulFloat32 computes C[M,N] = A[M,K] * B[K,N] from host slices.
func MatMulFloat32(M, N, K int, A, B, C []float32) error {
	empty, err := checkHostMatMul("matmul_f32", M, N, K, len(A), len(B), len(C))
	if err != nil {
		return err
	}
	if empty {
		return nil
	}
	dA, err := AllocF32(M * K)
	if err != nil {
		return err
	}
	defer dA.Free()
	dB, err := AllocF32(K * N)
	if err != nil {
		return err
	}
	defer dB.Free()
	dC, err := AllocF32(M * N)
	if err != nil {
		return err
	}
	defer dC.Free()
	if err := dA.Upload(A); err != nil {
		return err
	}
	if err := dB.Upload(B); err != nil {
		return err
	}
	if err := GemmF32(dA, dB, dC, M, N, K); err != nil {
		return err
	}
	return dC.Download(C)
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
	dA, err := AllocF64(M * K)
	if err != nil {
		return err
	}
	defer dA.Free()
	dB, err := AllocF64(K * N)
	if err != nil {
		return err
	}
	defer dB.Free()
	dC, err := AllocF64(M * N)
	if err != nil {
		return err
	}
	defer dC.Free()
	if err := dA.UploadF64(A); err != nil {
		return err
	}
	if err := dB.UploadF64(B); err != nil {
		return err
	}
	if err := GemmF64(dA, dB, dC, M, N, K); err != nil {
		return err
	}
	return dC.DownloadF64(C)
}

func elemwiseHost(what string, op int, a, b, out []float32) error {
	if err := checkHostOperands(what, len(out), len(a), len(b)); err != nil {
		return err
	}
	n := len(out)
	if n == 0 {
		return nil
	}
	dA, err := AllocF32(n)
	if err != nil {
		return err
	}
	defer dA.Free()
	dB, err := AllocF32(n)
	if err != nil {
		return err
	}
	defer dB.Free()
	dO, err := AllocF32(n)
	if err != nil {
		return err
	}
	defer dO.Free()
	if err := dA.Upload(a); err != nil {
		return err
	}
	if err := dB.Upload(b); err != nil {
		return err
	}
	if err := ElemwiseF32(op, dA, dB, dO, n); err != nil {
		return err
	}
	return dO.Download(out)
}

func elemwiseHost64(what string, op int, a, b, out []float64) error {
	if err := checkHostOperands(what, len(out), len(a), len(b)); err != nil {
		return err
	}
	n := len(out)
	if n == 0 {
		return nil
	}
	dA, err := AllocF64(n)
	if err != nil {
		return err
	}
	defer dA.Free()
	dB, err := AllocF64(n)
	if err != nil {
		return err
	}
	defer dB.Free()
	dO, err := AllocF64(n)
	if err != nil {
		return err
	}
	defer dO.Free()
	if err := dA.UploadF64(a); err != nil {
		return err
	}
	if err := dB.UploadF64(b); err != nil {
		return err
	}
	if err := ElemwiseF64(op, dA, dB, dO, n); err != nil {
		return err
	}
	return dO.DownloadF64(out)
}

// AddFloat32, SubFloat32, MulFloat32, DivFloat32 compute out=a OP b on host slices.
func AddFloat32(a, b, out []float32) error { return elemwiseHost("add_f32", OpAdd, a, b, out) }
func SubFloat32(a, b, out []float32) error { return elemwiseHost("sub_f32", OpSub, a, b, out) }
func MulFloat32(a, b, out []float32) error { return elemwiseHost("mul_f32", OpMul, a, b, out) }
func DivFloat32(a, b, out []float32) error { return elemwiseHost("div_f32", OpDiv, a, b, out) }

// AddFloat64, SubFloat64, MulFloat64, DivFloat64 compute out=a OP b on host slices.
func AddFloat64(a, b, out []float64) error { return elemwiseHost64("add_f64", OpAdd, a, b, out) }
func SubFloat64(a, b, out []float64) error { return elemwiseHost64("sub_f64", OpSub, a, b, out) }
func MulFloat64(a, b, out []float64) error { return elemwiseHost64("mul_f64", OpMul, a, b, out) }
func DivFloat64(a, b, out []float64) error { return elemwiseHost64("div_f64", OpDiv, a, b, out) }

// ScaleFloat32 computes out[i] = src[i] * alpha on host slices.
func ScaleFloat32(src []float32, alpha float32, out []float32) error {
	if err := checkHostOperands("scale_f32", len(out), len(src)); err != nil {
		return err
	}
	n := len(out)
	if n == 0 {
		return nil
	}
	dS, err := AllocF32(n)
	if err != nil {
		return err
	}
	defer dS.Free()
	dO, err := AllocF32(n)
	if err != nil {
		return err
	}
	defer dO.Free()
	if err := dS.Upload(src); err != nil {
		return err
	}
	if err := ScaleF32(dS, alpha, dO, n); err != nil {
		return err
	}
	return dO.Download(out)
}

// ScaleFloat64 computes out[i] = src[i] * alpha on host slices.
func ScaleFloat64(src []float64, alpha float64, out []float64) error {
	if err := checkHostOperands("scale_f64", len(out), len(src)); err != nil {
		return err
	}
	n := len(out)
	if n == 0 {
		return nil
	}
	dS, err := AllocF64(n)
	if err != nil {
		return err
	}
	defer dS.Free()
	dO, err := AllocF64(n)
	if err != nil {
		return err
	}
	defer dO.Free()
	if err := dS.UploadF64(src); err != nil {
		return err
	}
	if err := ScaleF64(dS, alpha, dO, n); err != nil {
		return err
	}
	return dO.DownloadF64(out)
}

// SumFloat32 reduces a host float32 slice on the GPU and returns the result.
// An empty slice is a documented no-op returning 0.
func SumFloat32(src []float32) (float32, error) {
	if len(src) == 0 {
		return 0, nil
	}
	dS, err := AllocF32(len(src))
	if err != nil {
		return 0, err
	}
	defer dS.Free()
	dO, err := AllocF32(1)
	if err != nil {
		return 0, err
	}
	defer dO.Free()
	if err := dS.Upload(src); err != nil {
		return 0, err
	}
	if err := SumF32(dS, dO, len(src)); err != nil {
		return 0, err
	}
	var out [1]float32
	if err := dO.Download(out[:]); err != nil {
		return 0, err
	}
	return out[0], nil
}

// MaxFloat32 reduces a host float32 slice on the GPU and returns the result.
// An empty slice is a documented no-op returning 0.
func MaxFloat32(src []float32) (float32, error) {
	if len(src) == 0 {
		return 0, nil
	}
	dS, err := AllocF32(len(src))
	if err != nil {
		return 0, err
	}
	defer dS.Free()
	dO, err := AllocF32(1)
	if err != nil {
		return 0, err
	}
	defer dO.Free()
	if err := dS.Upload(src); err != nil {
		return 0, err
	}
	if err := MaxF32(dS, dO, len(src)); err != nil {
		return 0, err
	}
	var bits [1]float32
	if err := dO.Download(bits[:]); err != nil {
		return 0, err
	}
	return DecodeMaxKeyF32(math.Float32bits(bits[0])), nil
}

// SumFloat64 reduces a host float64 slice on the GPU and returns the result.
// An empty slice is a documented no-op returning 0.
func SumFloat64(src []float64) (float64, error) {
	if len(src) == 0 {
		return 0, nil
	}
	dS, err := AllocF64(len(src))
	if err != nil {
		return 0, err
	}
	defer dS.Free()
	dO, err := AllocF64(1)
	if err != nil {
		return 0, err
	}
	defer dO.Free()
	if err := dS.UploadF64(src); err != nil {
		return 0, err
	}
	if err := SumF64(dS, dO, len(src)); err != nil {
		return 0, err
	}
	var out [1]float64
	if err := dO.DownloadF64(out[:]); err != nil {
		return 0, err
	}
	return out[0], nil
}

// MaxFloat64 reduces a host float64 slice on the GPU and returns the result.
// An empty slice is a documented no-op returning 0.
func MaxFloat64(src []float64) (float64, error) {
	if len(src) == 0 {
		return 0, nil
	}
	dS, err := AllocF64(len(src))
	if err != nil {
		return 0, err
	}
	defer dS.Free()
	dO, err := AllocF64(1)
	if err != nil {
		return 0, err
	}
	defer dO.Free()
	if err := dS.UploadF64(src); err != nil {
		return 0, err
	}
	if err := MaxF64(dS, dO, len(src)); err != nil {
		return 0, err
	}
	var out [1]float64
	if err := dO.DownloadF64(out[:]); err != nil {
		return 0, err
	}
	return out[0], nil
}

// ---------------------------------------------------------------------------
// Device-resident transformer/lm primitives (all float32).
// ---------------------------------------------------------------------------

// RMSNormF32 computes out = rmsnorm(x, gamma) for `rows` rows of `width`.
func RMSNormF32(x, gamma, out *Buffer, rows, width int, eps float32) error {
	if err := validateRMSNorm("rmsnorm_f32", x, gamma, out, rows, width); err != nil {
		return err
	}
	return launch(C.m3tal_launch_rmsnorm_f32(x.ptr, gamma.ptr, out.ptr,
		C.int(rows), C.int(width), C.float(eps)), "rmsnorm_f32 launch")
}

// RoPEBatchF32 applies rotary embeddings to q,k in place for `rows` rows.
func RoPEBatchF32(q, k *Buffer, rows, width, headDim, heads int) error {
	if err := validateRoPEBatch("rope_batch", q, k, rows, width, headDim, heads); err != nil {
		return err
	}
	return launch(C.m3tal_launch_rope_batch_f32(q.ptr, k.ptr,
		C.int(rows), C.int(width), C.int(headDim), C.int(heads)), "rope_batch launch")
}

// RoPESingleF32 applies rotary embeddings to a single row at position pos.
func RoPESingleF32(q, k *Buffer, pos, width, headDim, heads int) error {
	if err := validateRoPESingle("rope_single", q, k, pos, width, headDim, heads); err != nil {
		return err
	}
	return launch(C.m3tal_launch_rope_single_f32(q.ptr, k.ptr, C.int(pos),
		C.int(width), C.int(headDim), C.int(heads)), "rope_single launch")
}

// SwiGLUF32 computes out[i] = silu(gate[i]) * up[i].
func SwiGLUF32(gate, up, out *Buffer, n int) error {
	if err := validateSwiGLU("swiglu", gate, up, out, n); err != nil {
		return err
	}
	return launch(C.m3tal_launch_swiglu_f32(gate.ptr, up.ptr, out.ptr,
		C.longlong(n)), "swiglu launch")
}

// BiasAddF32 adds bias[j] to every row of C[M,N] in place.
func BiasAddF32(c, bias *Buffer, M, N int) error {
	if err := validateBiasAdd("bias_add", c, bias, M, N); err != nil {
		return err
	}
	return launch(C.m3tal_launch_bias_add_f32(c.ptr, bias.ptr,
		C.int(M), C.int(N)), "bias_add launch")
}

// EmbedF32 gathers rows: out[r,:] = emb[ids[r],:]. ids is an int32 buffer.
func EmbedF32(emb, ids, out *Buffer, rows, width int) error {
	if err := validateEmbed("embed", emb, ids, out, rows, width); err != nil {
		return err
	}
	return launch(C.m3tal_launch_embed_f32(emb.ptr, ids.ptr, out.ptr,
		C.int(rows), C.int(width)), "embed launch")
}

// AttnScoresF32 computes raw attention scores into attn[heads,qRows,kvLen].
// causal=1 masks j>i (batch prefill); causal=0 attends to all kvLen (decode).
func AttnScoresF32(q, k, attn *Buffer, heads, qRows, kvLen, width, headDim int,
	scale float32, causal bool) error {
	if err := validateAttnScores("attn_scores", q, k, attn, heads, qRows, kvLen, width, headDim); err != nil {
		return err
	}
	c := C.int(0)
	if causal {
		c = 1
	}
	return launch(C.m3tal_launch_attn_scores_f32(q.ptr, k.ptr, attn.ptr,
		C.int(heads), C.int(qRows), C.int(kvLen), C.int(width), C.int(headDim),
		C.float(scale), c), "attn_scores launch")
}

// AttnSoftmaxF32 normalizes each (head,row) attention row in place.
func AttnSoftmaxF32(attn *Buffer, heads, qRows, kvLen int) error {
	if err := validateAttnSoftmax("attn_softmax", attn, heads, qRows, kvLen); err != nil {
		return err
	}
	return launch(C.m3tal_launch_attn_softmax_f32(attn.ptr,
		C.int(heads), C.int(qRows), C.int(kvLen)), "attn_softmax launch")
}

// AttnWeightedF32 computes out = attn @ V per head into [qRows,width].
func AttnWeightedF32(attn, v, out *Buffer, heads, qRows, kvLen, width, headDim int) error {
	if err := validateAttnWeighted("attn_weighted", attn, v, out, heads, qRows, kvLen, width, headDim); err != nil {
		return err
	}
	return launch(C.m3tal_launch_attn_weighted_f32(attn.ptr, v.ptr, out.ptr,
		C.int(heads), C.int(qRows), C.int(kvLen), C.int(width), C.int(headDim)),
		"attn_weighted launch")
}
