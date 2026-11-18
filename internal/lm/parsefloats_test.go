package lm

import (
	"strconv"
	"strings"
	"testing"
)

func formatFloatArray(values []float32, separator string) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, v := range values {
		if i > 0 {
			b.WriteString(separator)
		}
		b.WriteString(strconv.FormatFloat(float64(v), 'g', -1, 32))
	}
	b.WriteByte(']')
	return b.String()
}

func assertParsed(t *testing.T, raw string, want []float32) {
	t.Helper()
	got := make([]float32, len(want))
	if err := parseFloatSliceInto([]byte(raw), got); err != nil {
		t.Fatalf("parse: %v", err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("value %d: %v != %v", i, got[i], want[i])
		}
	}
}

// TestParseFloatSliceLargeArrays exercises the parallel path, which only runs
// for arrays of at least 1<<15 values, in several formats.
func TestParseFloatSliceLargeArrays(t *testing.T) {
	values := make([]float32, 200000)
	for i := range values {
		values[i] = float32(i%1000)*0.25 - 7.5
	}
	assertParsed(t, formatFloatArray(values, ","), values)
	assertParsed(t, formatFloatArray(values, ","), values)

	// Pretty-printed input: whitespace around separators must be skipped.
	joined := formatFloatArray(values, ",\n  ")
	joined = strings.TrimSuffix(strings.TrimPrefix(joined, "["), "]")
	assertParsed(t, "[\n  "+joined+"\n]", values)

	// Single value and empty arrays use the serial path.
	assertParsed(t, "[1.5]", []float32{1.5})
	assertParsed(t, "[]", nil)

	// Count mismatches must be reported, not silently truncated or padded.
	for _, tc := range []struct {
		raw  string
		size int
	}{
		{formatFloatArray(values, ","), len(values) - 1},
		{formatFloatArray(values, ","), len(values) + 1},
	} {
		target := make([]float32, tc.size)
		if err := parseFloatSliceInto([]byte(tc.raw), target); err == nil {
			t.Fatalf("expected error for %d values into %d slots", len(values), tc.size)
		}
	}

	// Malformed numbers must surface the offending text.
	if err := parseFloatSliceInto([]byte("[1,2,oops]"), make([]float32, 3)); err == nil {
		t.Fatal("expected error for malformed number")
	}
	if err := parseFloatSliceInto([]byte("not an array"), make([]float32, 1)); err == nil {
		t.Fatal("expected error for non-array input")
	}
}
