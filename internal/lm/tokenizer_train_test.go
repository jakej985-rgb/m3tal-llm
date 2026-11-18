package lm

import (
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"
)

// trainTokenizerReference is the original training loop, kept verbatim as the
// oracle: rescan every pair of every sequence per merge, sort all distinct
// pairs, apply the merge with applyMerge. trainMerges must reproduce it
// exactly, including tie-breaking.
func trainTokenizerReference(corpus []string, cfg TokenizerConfig) (*Tokenizer, error) {
	targetSize := cfg.VocabSize
	t := &Tokenizer{
		Tokens:        make([]string, 0, targetSize),
		SpecialTokens: make(map[string]int),
		unkTokenID:    -1,
	}
	known := make(map[string]bool, targetSize)
	for i := 0; i < 256; i++ {
		bTok := string([]byte{byte(i)})
		t.Tokens = append(t.Tokens, bTok)
		known[bTok] = true
	}
	if len(cfg.SpecialTokens) > 0 {
		stNames := make([]string, 0, len(cfg.SpecialTokens))
		for st := range cfg.SpecialTokens {
			stNames = append(stNames, st)
		}
		sort.Strings(stNames)
		for _, st := range stNames {
			if !known[st] {
				id := len(t.Tokens)
				t.Tokens = append(t.Tokens, st)
				known[st] = true
				t.SpecialTokens[st] = id
			} else {
				t.SpecialTokens[st] = t.indexOf(st)
			}
		}
	}
	if cfg.UnknownToken != "" {
		if id, ok := t.SpecialTokens[cfg.UnknownToken]; ok {
			t.unkTokenID = id
		}
	}

	sequences := make([][]string, 0, len(corpus))
	for _, text := range corpus {
		for _, chunk := range t.splitSpecialTokens(text) {
			if !chunk.isSpecial && len(chunk.text) > 0 {
				sequences = append(sequences, byteSymbols(chunk.text))
			}
		}
	}

	// The pair itself is the map key: a joined string would be ambiguous
	// because the vocabulary contains every byte, including 0x00.
	type refPair struct{ left, right string }
	for len(t.Tokens) < targetSize {
		countOf := make(map[refPair]int)
		for _, seq := range sequences {
			for i := 0; i+1 < len(seq); i++ {
				if !known[seq[i]+seq[i+1]] {
					countOf[refPair{seq[i], seq[i+1]}]++
				}
			}
		}
		keys := make([]refPair, 0, len(countOf))
		for p, count := range countOf {
			if count > 1 {
				keys = append(keys, p)
			}
		}
		if len(keys) == 0 {
			break
		}
		sort.Slice(keys, func(i, j int) bool {
			if keys[i].left == keys[j].left {
				return keys[i].right < keys[j].right
			}
			return keys[i].left < keys[j].left
		})
		maxCount := 0
		for _, candidate := range keys {
			if countOf[candidate] > maxCount {
				maxCount = countOf[candidate]
			}
		}
		best := keys[0]
		for _, candidate := range keys {
			if countOf[candidate] == maxCount {
				best = candidate
				break
			}
		}
		merged := best.left + best.right
		known[merged] = true
		t.Merges = append(t.Merges, Merge{Left: best.left, Right: best.right, Token: merged})
		t.Tokens = append(t.Tokens, merged)
		for i, seq := range sequences {
			sequences[i] = applyMerge(seq, best.left, best.right, merged)
		}
	}
	t.reindex()
	return t, nil
}

func assertSameVocab(t *testing.T, label string, got, want *Tokenizer) {
	t.Helper()
	if len(got.Tokens) != len(want.Tokens) {
		t.Fatalf("%s: vocab size %d, want %d", label, len(got.Tokens), len(want.Tokens))
	}
	for i := range want.Tokens {
		if got.Tokens[i] != want.Tokens[i] {
			t.Fatalf("%s: token %d = %q, want %q", label, i, got.Tokens[i], want.Tokens[i])
		}
	}
	if len(got.Merges) != len(want.Merges) {
		t.Fatalf("%s: %d merges, want %d", label, len(got.Merges), len(want.Merges))
	}
	for i := range want.Merges {
		if got.Merges[i] != want.Merges[i] {
			t.Fatalf("%s: merge %d = %+v, want %+v", label, i, got.Merges[i], want.Merges[i])
		}
	}
	if got.unkTokenID != want.unkTokenID {
		t.Fatalf("%s: unk id %d, want %d", label, got.unkTokenID, want.unkTokenID)
	}
	if len(got.SpecialTokens) != len(want.SpecialTokens) {
		t.Fatalf("%s: %d special tokens, want %d", label, len(got.SpecialTokens), len(want.SpecialTokens))
	}
	for k, v := range want.SpecialTokens {
		if got.SpecialTokens[k] != v {
			t.Fatalf("%s: special token %q = %d, want %d", label, k, got.SpecialTokens[k], v)
		}
	}
}

func randomCorpus(rng *rand.Rand, lines, length int) []string {
	alphabets := []string{
		"ab",
		"abc ",
		"aabbc\n",
		"the quick brown fox. \n",
		"é漢🙂 ab \n",
	}
	out := make([]string, lines)
	for i := range out {
		alphabet := []rune(alphabets[rng.Intn(len(alphabets))])
		var b strings.Builder
		for j := 0; j < length; j++ {
			b.WriteRune(alphabet[rng.Intn(len(alphabet))])
		}
		out[i] = b.String()
	}
	return out
}

func TestTrainTokenizerMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	realCorpus := func() []string {
		data, err := os.ReadFile("../../data/curriculum-train.txt")
		if err != nil {
			return nil
		}
		if len(data) > 8000 {
			data = data[:8000]
		}
		return []string{string(data)}
	}()

	cases := []struct {
		name   string
		corpus []string
		cfg    TokenizerConfig
	}{
		{"no merges", []string{"abab abab"}, TokenizerConfig{VocabSize: 256}},
		{"single merge", []string{"abababab"}, TokenizerConfig{VocabSize: 257}},
		{"repetitive", []string{"aabbaabb", "aaaa", "abab", "abcabc"}, TokenizerConfig{VocabSize: 300}},
		{"ties", []string{strings.Repeat("ab", 50), strings.Repeat("cd", 50)}, TokenizerConfig{VocabSize: 280}},
		{"random ab", randomCorpus(rng, 6, 400), TokenizerConfig{VocabSize: 400}},
		{"random abc", randomCorpus(rng, 6, 400), TokenizerConfig{VocabSize: 400}},
		{"random words", randomCorpus(rng, 8, 300), TokenizerConfig{VocabSize: 512}},
		{"random utf8", randomCorpus(rng, 6, 200), TokenizerConfig{VocabSize: 420}},
		{"special tokens", []string{
			"<|im_start|>user\nhello there<|im_end|>\n<|im_start|>assistant\nhi<|im_end|>\n",
			"hello there hello there",
		}, TokenizerConfig{
			VocabSize:     320,
			SpecialTokens: map[string]int{"<|im_start|>": 0, "<|im_end|>": 0, "<|pad|>": 0},
			UnknownToken:  "<|pad|>",
		}},
		{"ambiguous splits", []string{"abcabcabc", "aabbcc", "abcabc"}, TokenizerConfig{VocabSize: 300}},
		{"nul bytes", []string{"a\x00ba\x00b\x00a", "b\x00a\x00b", strings.Repeat("\x00a\x00b", 20)}, TokenizerConfig{VocabSize: 300}},
		{"real corpus", realCorpus, TokenizerConfig{VocabSize: 480}},
	}

	for _, tc := range cases {
		if tc.corpus == nil {
			continue
		}
		t.Run(tc.name, func(t *testing.T) {
			got, err := TrainTokenizerWithOptions(tc.corpus, tc.cfg)
			if err != nil {
				t.Fatal(err)
			}
			want, err := trainTokenizerReference(tc.corpus, tc.cfg)
			if err != nil {
				t.Fatal(err)
			}
			assertSameVocab(t, tc.name, got, want)
			// A trained tokenizer must also encode identically, since the
			// reference encoder is what artifacts were built with.
			for _, text := range tc.corpus {
				if a, b := got.Encode(text), encodeReference(want, text); len(a) != len(b) {
					t.Fatalf("%s: encode mismatch for %q: %d vs %d tokens", tc.name, text, len(a), len(b))
				}
			}
		})
	}
}

func benchTrainCorpus(b *testing.B, limit int) []string {
	b.Helper()
	data, err := os.ReadFile("../../data/curriculum-train.txt")
	if err != nil {
		b.Skip(err)
	}
	if len(data) > limit {
		data = data[:limit]
	}
	return []string{string(data)}
}

func BenchmarkTrainTokenizer512(b *testing.B) {
	corpus := benchTrainCorpus(b, 200_000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := TrainTokenizer(corpus, 512); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTrainTokenizer1536(b *testing.B) {
	corpus := benchTrainCorpus(b, 200_000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := TrainTokenizer(corpus, 1536); err != nil {
			b.Fatal(err)
		}
	}
}
