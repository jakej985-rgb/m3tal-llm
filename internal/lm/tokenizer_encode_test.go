package lm

import (
	"math/rand"
	"os"
	"strings"
	"testing"
)

// encodeReference is the original encoder: replay every merge in learned
// order over the whole symbol sequence, left to right, allocating a new
// sequence per merge. The fast path in Encode must match it exactly,
// including special-token handling.
func encodeReference(t *Tokenizer, text string) []int {
	var ids []int
	for _, chunk := range t.splitSpecialTokens(text) {
		if chunk.isSpecial {
			ids = append(ids, chunk.id)
			continue
		}
		seq := byteSymbols(chunk.text)
		for _, merge := range t.Merges {
			seq = applyMerge(seq, merge.Left, merge.Right, merge.Token)
		}
		for _, token := range seq {
			if id, ok := t.index[token]; ok {
				ids = append(ids, id)
			} else if t.unkTokenID >= 0 {
				ids = append(ids, t.unkTokenID)
			}
		}
	}
	return ids
}

func randomText(rng *rand.Rand, length int) string {
	alphabet := []rune("ab abc \n{}:<|im_start|><|im_end|> éü漢字🙂")
	var b strings.Builder
	for i := 0; i < length; i++ {
		b.WriteRune(alphabet[rng.Intn(len(alphabet))])
	}
	return b.String()
}

func assertEncodeMatches(t *testing.T, tok *Tokenizer, text string) {
	t.Helper()
	got := tok.Encode(text)
	want := encodeReference(tok, text)
	if len(got) != len(want) {
		t.Fatalf("token count: fast=%d reference=%d for %q", len(got), len(want), text)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: fast=%d reference=%d for %q", i, got[i], want[i], text)
		}
	}
}

// assertFastPathActive guards the differential tests: if a tokenizer fell back
// to encodeChunkSlow they would compare the slow path with itself.
func assertFastPathActive(t *testing.T, tok *Tokenizer) {
	t.Helper()
	if !tok.hasByteIDs {
		t.Fatal("tokenizer has no byte-token table; comparisons would not exercise the fast path")
	}
}

func TestEncodeMatchesReference(t *testing.T) {
	corpus := []string{
		strings.Repeat("the quick brown fox jumps over the lazy dog. ", 40),
		"<|im_start|>user\nExplain BPE.<|im_end|>\n<|im_start|>assistant\nSure<|im_end|>\n",
		"",
		"a",
		"ab",
		"aabbaabb",
		"ababababab",
		"aaaaaaaaaa",
		"ééé 漢字漢字 🙂🙂",
		"\xff\xfe invalid utf-8 \x00\x01\x02 control bytes",
	}
	for _, size := range []int{256, 300, 512, 700} {
		tok, err := TrainTokenizer(corpus, size)
		if err != nil {
			t.Fatalf("train tokenizer %d: %v", size, err)
		}
		assertFastPathActive(t, tok)
		for _, text := range corpus {
			assertEncodeMatches(t, tok, text)
		}
		// Overlapping merges are the risky case for a rank-ordered encoder:
		// repeated units create rules like (a,a)->aa and (aa,aa)->aaaa, and
		// prefixes cut sequences at every possible boundary.
		for _, unit := range []string{"a", "ab", "abc", "aab", "aaaa", "é"} {
			for k := 0; k <= 24; k++ {
				assertEncodeMatches(t, tok, strings.Repeat(unit, k))
			}
		}
		mixed := "abracadabra abcabc <|im_end|> é漢🙂 aabbaabb"
		for i := 0; i <= len(mixed); i++ {
			assertEncodeMatches(t, tok, mixed[:i])
		}
		rng := rand.New(rand.NewSource(int64(size)))
		for i := 0; i < 40; i++ {
			assertEncodeMatches(t, tok, randomText(rng, rng.Intn(64)))
		}
	}
}

// TestEncodeDuplicateMergeRules covers a malformed artifact that keeps the
// same pair twice: the reference applies the first rule (which removes every
// occurrence) and finds nothing for the second, so the fast path must keep
// the earliest rule rather than overwriting the rank.
func TestEncodeDuplicateMergeRules(t *testing.T) {
	tok := &Tokenizer{SpecialTokens: map[string]int{}, unkTokenID: -1}
	for i := 0; i < 256; i++ {
		tok.Tokens = append(tok.Tokens, string([]byte{byte(i)}))
	}
	tok.Tokens = append(tok.Tokens, "ab")
	duplicate := Merge{Left: "a", Right: "b", Token: "ab"}
	tok.Merges = []Merge{duplicate, duplicate}
	tok.reindex()
	assertFastPathActive(t, tok)
	for _, text := range []string{"ab", "abab", "aabbab", "b", "aab"} {
		assertEncodeMatches(t, tok, text)
	}
	if got := tok.Encode("abab"); len(got) != 2 || got[0] != 256 || got[1] != 256 {
		t.Fatalf("duplicate-rule encode: got %v, want [256 256]", got)
	}
}

func TestEncodeMatchesReferenceRealTokenizer(t *testing.T) {
	const path = "../../models/tokenizer-1536.json"
	if _, err := os.Stat(path); err != nil {
		t.Skip("tokenizer artifact not present")
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	assertFastPathActive(t, tok)
	corpus := []string{
		"The quick brown fox jumps over the lazy dog. ",
		"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>\n",
		"func main() { fmt.Println(\"hello\") }\n",
		"éü漢字🙂 mixed bytes",
	}
	for _, text := range corpus {
		assertEncodeMatches(t, tok, strings.Repeat(text, 8))
	}
	for _, unit := range []string{"a", "the ", "\n\n", "e"} {
		for k := 0; k <= 40; k++ {
			assertEncodeMatches(t, tok, strings.Repeat(unit, k))
		}
	}
	rng := rand.New(rand.NewSource(7))
	for i := 0; i < 50; i++ {
		assertEncodeMatches(t, tok, randomText(rng, rng.Intn(400)))
	}
}

func BenchmarkEncode1KB(b *testing.B) {
	tok, err := LoadTokenizer("../../models/tokenizer-1536.json")
	if err != nil {
		b.Skip("tokenizer artifact not present")
	}
	text := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 24)
	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok.Encode(text)
	}
}
