package gpu

// Boundary validation for device operations.
//
// CUDA kernels receive raw device pointers, so a wrong offset, length or
// dimension is not a recoverable error at the device level: it silently
// reads or writes memory outside the allocation. Every public operation in
// this package therefore validates its buffers, offsets, lengths and shapes
// here, before any cudaMalloc, cudaMemcpy or kernel launch, and returns an
// error wrapping ErrInvalid instead of dispatching.
//
// These checks are compiled into both the CUDA and the stub build, so callers
// see the same validation errors with and without a device.

import (
	"errors"
	"fmt"
	"math"
)

// ErrInvalid reports a rejected device operation: a nil or freed buffer, an
// empty buffer where data is required, a negative, undersized or overflowing
// length, an out-of-range offset, a shape mismatch, or buffers belonging to
// different devices. When an error wraps ErrInvalid nothing was allocated,
// copied or launched.
var ErrInvalid = errors.New("invalid gpu operation")

func invalidf(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrInvalid, fmt.Sprintf(format, args...))
}

// elemBytes converts an element count into a byte count, rejecting negative
// counts and multiplication overflow.
func elemBytes(elems, size int, what string) (int, error) {
	if elems < 0 {
		return 0, invalidf("%s: negative element count %d", what, elems)
	}
	if size <= 0 {
		return 0, invalidf("%s: invalid element size %d", what, size)
	}
	if elems > math.MaxInt/size {
		return 0, invalidf("%s: %d elements of %d bytes overflow the byte count",
			what, elems, size)
	}
	return elems * size, nil
}

// dimBytes converts the product of dims, in elements of size bytes, into a
// byte count. Every dimension must be positive and the product must not
// overflow.
func dimBytes(size int, what string, dims ...int) (int, error) {
	total := 1
	for _, d := range dims {
		if d <= 0 {
			return 0, invalidf("%s: non-positive dimension %d (dims %v)", what, d, dims)
		}
		if total > math.MaxInt/d {
			return 0, invalidf("%s: dimension product %v overflows", what, dims)
		}
		total *= d
	}
	return elemBytes(total, size, what)
}

// checkSpan validates the byte range [off, off+n) inside a size-byte
// allocation. It compares against size-off instead of computing off+n so a
// hostile offset cannot wrap around.
func checkSpan(size, off, n int, what string) error {
	if off < 0 {
		return invalidf("%s: negative offset %d", what, off)
	}
	if n < 0 {
		return invalidf("%s: negative length %d", what, n)
	}
	if off > size {
		return invalidf("%s: offset %d is past the end of a %d-byte buffer",
			what, off, size)
	}
	if n > size-off {
		return invalidf("%s: %d bytes at offset %d exceed the %d-byte buffer",
			what, n, off, size)
	}
	return nil
}

// usable rejects nil and freed buffers and any set of buffers allocated on
// different devices. Every device operation calls it before touching memory.
func usable(what string, buffers ...*Buffer) error {
	for i, b := range buffers {
		if b == nil {
			return invalidf("%s: buffer %d is nil", what, i)
		}
		if b.freed {
			return invalidf("%s: buffer %d (%d bytes) was already freed",
				what, i, b.bytes)
		}
	}
	for i := 1; i < len(buffers); i++ {
		if buffers[i].device != buffers[0].device {
			return invalidf("%s: buffer %d is on device %d but buffer 0 is on device %d",
				what, i, buffers[i].device, buffers[0].device)
		}
	}
	return nil
}

// fit is one buffer's byte requirement, named for error messages.
type fit struct {
	name  string
	buf   *Buffer
	bytes int
}

// require checks that every buffer in fits holds at least its required bytes.
// Callers must have accepted the buffers with usable first.
func require(what string, fits ...fit) error {
	for _, f := range fits {
		if f.buf.bytes < f.bytes {
			return invalidf("%s: %s needs %d bytes, buffer holds %d",
				what, f.name, f.bytes, f.buf.bytes)
		}
	}
	return nil
}

// checkHeads validates a multi-head shape: heads*headDim must equal width.
func checkHeads(what string, width, headDim, heads int) error {
	if width <= 0 || headDim <= 0 || heads <= 0 {
		return invalidf("%s: non-positive width=%d headDim=%d heads=%d",
			what, width, headDim, heads)
	}
	if heads > width || width%heads != 0 || width/heads != headDim {
		return invalidf("%s: width %d is not heads(%d) * headDim(%d)",
			what, width, heads, headDim)
	}
	return nil
}

// checkTransfer validates a host<->device copy of n bytes against b. Copies
// must fill the buffer exactly, so a short slice can never leave part of the
// allocation holding stale or uninitialized data.
func checkTransfer(what string, b *Buffer, n int) error {
	if err := usable(what, b); err != nil {
		return err
	}
	if n < 0 {
		return invalidf("%s: negative transfer length %d", what, n)
	}
	if n != b.bytes {
		return invalidf("%s: a %d-byte transfer does not match the %d-byte buffer (copies fill the buffer exactly)",
			what, n, b.bytes)
	}
	return nil
}

// validateGemm checks A[M,K] * B[K,N] -> Out[M,N].
func validateGemm(what string, A, B, Out *Buffer, size, M, N, K int) error {
	if err := usable(what, A, B, Out); err != nil {
		return err
	}
	ak, err := dimBytes(size, what+": lhs", M, K)
	if err != nil {
		return err
	}
	kn, err := dimBytes(size, what+": rhs", K, N)
	if err != nil {
		return err
	}
	mn, err := dimBytes(size, what+": out", M, N)
	if err != nil {
		return err
	}
	return require(what, fit{"lhs", A, ak}, fit{"rhs", B, kn}, fit{"out", Out, mn})
}

// validateElemwise checks out[i] = a[i] OP b[i] for n elements.
func validateElemwise(what string, A, B, Out *Buffer, size, n int) error {
	if err := usable(what, A, B, Out); err != nil {
		return err
	}
	nb, err := dimBytes(size, what, n)
	if err != nil {
		return err
	}
	return require(what, fit{"a", A, nb}, fit{"b", B, nb}, fit{"out", Out, nb})
}

// validateScale checks out[i] = src[i] * alpha for n elements.
func validateScale(what string, Src, Out *Buffer, size, n int) error {
	if err := usable(what, Src, Out); err != nil {
		return err
	}
	nb, err := dimBytes(size, what, n)
	if err != nil {
		return err
	}
	return require(what, fit{"src", Src, nb}, fit{"out", Out, nb})
}

// validateReduce checks a reduction of n source elements into one element.
func validateReduce(what string, Src, Out *Buffer, size, n int) error {
	if err := usable(what, Src, Out); err != nil {
		return err
	}
	nb, err := dimBytes(size, what+": src", n)
	if err != nil {
		return err
	}
	outb, err := dimBytes(size, what+": out", 1)
	if err != nil {
		return err
	}
	return require(what, fit{"src", Src, nb}, fit{"out", Out, outb})
}

// validateMemcpyD2D checks a device-to-device copy of n bytes at the given
// byte offsets. Overlapping ranges inside one buffer are rejected because
// cudaMemcpy is undefined for them.
func validateMemcpyD2D(what string, dst *Buffer, dstOff int, src *Buffer, srcOff, n int) error {
	if err := usable(what, dst, src); err != nil {
		return err
	}
	if err := checkSpan(dst.bytes, dstOff, n, what+": dst"); err != nil {
		return err
	}
	if err := checkSpan(src.bytes, srcOff, n, what+": src"); err != nil {
		return err
	}
	if dst == src && n > 0 && dstOff != srcOff {
		return invalidf("%s: %d bytes overlap between offsets %d and %d; copy through a scratch buffer",
			what, n, srcOff, dstOff)
	}
	return nil
}

// validateRMSNorm checks out = rmsnorm(x, gamma) for rows of width.
func validateRMSNorm(what string, x, gamma, out *Buffer, rows, width int) error {
	if err := usable(what, x, gamma, out); err != nil {
		return err
	}
	rw, err := dimBytes(4, what+": x", rows, width)
	if err != nil {
		return err
	}
	g, err := dimBytes(4, what+": gamma", width)
	if err != nil {
		return err
	}
	return require(what, fit{"x", x, rw}, fit{"gamma", gamma, g}, fit{"out", out, rw})
}

// validateRoPEBatch checks in-place rotary embeddings for rows of width.
func validateRoPEBatch(what string, q, k *Buffer, rows, width, headDim, heads int) error {
	if err := usable(what, q, k); err != nil {
		return err
	}
	if err := checkHeads(what, width, headDim, heads); err != nil {
		return err
	}
	rw, err := dimBytes(4, what, rows, width)
	if err != nil {
		return err
	}
	return require(what, fit{"q", q, rw}, fit{"k", k, rw})
}

// validateRoPESingle checks in-place rotary embeddings for one row at pos.
func validateRoPESingle(what string, q, k *Buffer, pos, width, headDim, heads int) error {
	if err := usable(what, q, k); err != nil {
		return err
	}
	if err := checkHeads(what, width, headDim, heads); err != nil {
		return err
	}
	if pos < 0 {
		return invalidf("%s: negative position %d", what, pos)
	}
	w, err := dimBytes(4, what, width)
	if err != nil {
		return err
	}
	return require(what, fit{"q", q, w}, fit{"k", k, w})
}

// validateSwiGLU checks out[i] = silu(gate[i]) * up[i] for n elements.
func validateSwiGLU(what string, gate, up, out *Buffer, n int) error {
	if err := usable(what, gate, up, out); err != nil {
		return err
	}
	nb, err := dimBytes(4, what, n)
	if err != nil {
		return err
	}
	return require(what, fit{"gate", gate, nb}, fit{"up", up, nb}, fit{"out", out, nb})
}

// validateBiasAdd checks C[M,N] += bias[N].
func validateBiasAdd(what string, c, bias *Buffer, M, N int) error {
	if err := usable(what, c, bias); err != nil {
		return err
	}
	mn, err := dimBytes(4, what+": c", M, N)
	if err != nil {
		return err
	}
	n, err := dimBytes(4, what+": bias", N)
	if err != nil {
		return err
	}
	return require(what, fit{"c", c, mn}, fit{"bias", bias, n})
}

// validateEmbed checks the gather out[r,:] = emb[ids[r],:] for rows of width.
func validateEmbed(what string, emb, ids, out *Buffer, rows, width int) error {
	if err := usable(what, emb, ids, out); err != nil {
		return err
	}
	row, err := dimBytes(4, what+": width", width)
	if err != nil {
		return err
	}
	if emb.bytes%row != 0 {
		return invalidf("%s: %d-byte embedding table is not a whole number of %d-byte rows",
			what, emb.bytes, row)
	}
	idBytes, err := dimBytes(4, what+": ids", rows)
	if err != nil {
		return err
	}
	rb, err := dimBytes(4, what+": out", rows, width)
	if err != nil {
		return err
	}
	return require(what, fit{"emb", emb, row}, fit{"ids", ids, idBytes}, fit{"out", out, rb})
}

// validateAttnScores checks attn[heads,qRows,kvLen] = q @ k^T.
func validateAttnScores(what string, q, k, attn *Buffer, heads, qRows, kvLen, width, headDim int) error {
	if err := usable(what, q, k, attn); err != nil {
		return err
	}
	if err := checkHeads(what, width, headDim, heads); err != nil {
		return err
	}
	qb, err := dimBytes(4, what+": q", qRows, width)
	if err != nil {
		return err
	}
	kb, err := dimBytes(4, what+": k", kvLen, width)
	if err != nil {
		return err
	}
	ab, err := dimBytes(4, what+": attn", heads, qRows, kvLen)
	if err != nil {
		return err
	}
	return require(what, fit{"q", q, qb}, fit{"k", k, kb}, fit{"attn", attn, ab})
}

// validateAttnSoftmax checks the in-place normalization of attn[heads,qRows,kvLen].
func validateAttnSoftmax(what string, attn *Buffer, heads, qRows, kvLen int) error {
	if err := usable(what, attn); err != nil {
		return err
	}
	ab, err := dimBytes(4, what, heads, qRows, kvLen)
	if err != nil {
		return err
	}
	return require(what, fit{"attn", attn, ab})
}

// validateAttnWeighted checks out[qRows,width] = attn @ v per head.
func validateAttnWeighted(what string, attn, v, out *Buffer, heads, qRows, kvLen, width, headDim int) error {
	if err := usable(what, attn, v, out); err != nil {
		return err
	}
	if err := checkHeads(what, width, headDim, heads); err != nil {
		return err
	}
	ab, err := dimBytes(4, what+": attn", heads, qRows, kvLen)
	if err != nil {
		return err
	}
	vb, err := dimBytes(4, what+": v", kvLen, width)
	if err != nil {
		return err
	}
	ob, err := dimBytes(4, what+": out", qRows, width)
	if err != nil {
		return err
	}
	return require(what, fit{"attn", attn, ab}, fit{"v", v, vb}, fit{"out", out, ob})
}

// checkHostMatMul validates the shape and operand lengths of a host-slice
// matmul. It reports whether the shape is empty, which callers treat as a
// documented no-op.
func checkHostMatMul(what string, M, N, K, lenA, lenB, lenC int) (bool, error) {
	for _, d := range []int{M, N, K} {
		if d < 0 {
			return false, invalidf("%s: negative dimension in M=%d N=%d K=%d",
				what, M, N, K)
		}
	}
	if M == 0 || N == 0 || K == 0 {
		if lenA != 0 || lenB != 0 || lenC != 0 {
			return false, invalidf("%s: empty shape M=%d N=%d K=%d with non-empty operands (A=%d B=%d C=%d)",
				what, M, N, K, lenA, lenB, lenC)
		}
		return true, nil
	}
	ak, err := dimBytes(1, what+": A", M, K)
	if err != nil {
		return false, err
	}
	kn, err := dimBytes(1, what+": B", K, N)
	if err != nil {
		return false, err
	}
	mn, err := dimBytes(1, what+": C", M, N)
	if err != nil {
		return false, err
	}
	switch {
	case lenA != ak:
		return false, invalidf("%s: A has %d elements, want M*K=%d", what, lenA, ak)
	case lenB != kn:
		return false, invalidf("%s: B has %d elements, want K*N=%d", what, lenB, kn)
	case lenC != mn:
		return false, invalidf("%s: C has %d elements, want M*N=%d", what, lenC, mn)
	}
	return false, nil
}

// checkHostOperands validates that every input slice length matches the
// output length. A zero-length output requires zero-length inputs, which
// callers treat as a documented no-op.
func checkHostOperands(what string, outLen int, inputs ...int) error {
	for _, n := range inputs {
		if n != outLen {
			return invalidf("%s: operand length %d does not match output length %d",
				what, n, outLen)
		}
	}
	return nil
}
