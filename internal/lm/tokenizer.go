// Package lm implements a small language model and tokenizer in pure Go.
//
// The tokenizer implements byte-level Byte-Pair Encoding (BPE) with support
// for configurable vocabulary sizes from 256 up to 32,768 (32k baseline),
// deterministic training, special token handling (<|endoftext|>, <|pad|>, <|unk|>),
// and versioned serialization with metadata.
//
// Vocabulary sizes can be adjusted based on the training corpus and hardware
// constraints (e.g. smaller vocabularies such as 256–2048 for tiny toy datasets
// or specialized domains, and up to 32,768 for general conversational pretraining).
package lm

import (
	"container/heap"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	// MinVocabSize is the minimum allowable vocabulary size (256 raw bytes).
	MinVocabSize = 256
	// MaxVocabSize is the maximum allowable vocabulary size (32k baseline).
	MaxVocabSize = 32768
	// TokenizerArtifactVersion is the current version string for saved tokenizer artifacts.
	TokenizerArtifactVersion = "1.0.0"

	// Default special token string constants.
	DefaultEndOfTextToken = "<|endoftext|>"
	DefaultPadToken       = "<|pad|>"
	DefaultUnkToken       = "<|unk|>"
)

// Tokenizer is a byte-level BPE tokenizer. IDs 0..255 represent raw bytes,
// followed by optional special tokens, followed by learned merges appended in training order.
type Tokenizer struct {
	Tokens        []string       `json:"tokens"`
	Merges        []Merge        `json:"merges"`
	SpecialTokens map[string]int `json:"special_tokens,omitempty"`
	unkTokenID    int
	index         map[string]int

	// Encoding fast path, rebuilt by reindex: mergeIndex maps an adjacent
	// (left, right) token-id pair to its merge rule, and byteIDs maps a raw
	// byte to its token id. hasByteIDs is false for tokenizers that do not
	// start with the 256 byte tokens, which fall back to the reference path.
	mergeIndex map[uint64]bpeRule
	byteIDs    [256]int32
	hasByteIDs bool
}

// bpeRule is one learned merge: its rank (position in the merge list, i.e.
// learned order) and the token id it produces.
type bpeRule struct {
	rank uint32
	out  int32
}

func mergeKey(left, right int32) uint64 {
	return uint64(uint32(left))<<32 | uint64(uint32(right))
}

// bpeQueue is a min-heap of packed (rank, position) pairs. Ordering by rank
// then position reproduces "replay merges in learned order, left to right".
type bpeQueue []uint64

func (q bpeQueue) Len() int           { return len(q) }
func (q bpeQueue) Less(i, j int) bool { return q[i] < q[j] }
func (q bpeQueue) Swap(i, j int)      { q[i], q[j] = q[j], q[i] }
func (q *bpeQueue) Push(x any)        { *q = append(*q, x.(uint64)) }
func (q *bpeQueue) Pop() any {
	old := *q
	n := len(old)
	item := old[n-1]
	*q = old[:n-1]
	return item
}

func packPair(rank uint32, pos int32) uint64 {
	return uint64(rank)<<32 | uint64(uint32(pos))
}

// Merge records one ordered byte-pair merge.
type Merge struct {
	Left  string `json:"left"`
	Right string `json:"right"`
	Token string `json:"token"`
}

// TokenizerConfig configures tokenizer training and special token assignments.
// Target vocabulary size can be adjusted based on the target corpus size.
type TokenizerConfig struct {
	VocabSize     int            `json:"vocab_size"`
	SpecialTokens map[string]int `json:"special_tokens,omitempty"`
	UnknownToken  string         `json:"unknown_token,omitempty"`
}

// TokenizerMetadata stores versioning and configuration metadata with saved tokenizer artifacts.
type TokenizerMetadata struct {
	Version   string          `json:"version"`
	VocabSize int             `json:"vocab_size"`
	Config    TokenizerConfig `json:"config"`
}

// TokenizerArtifact is the complete serialized representation of a trained tokenizer.
// Tokens and merges use base64-encoded byte slices for robust byte-level serialization.
type TokenizerArtifact struct {
	Metadata      TokenizerMetadata `json:"metadata"`
	Tokens        [][]byte          `json:"tokens"`
	Merges        []artifactMerge   `json:"merges"`
	SpecialTokens map[string]int    `json:"special_tokens,omitempty"`
}

type artifactMerge struct {
	Left  []byte `json:"left"`
	Right []byte `json:"right"`
	Token []byte `json:"token"`
}

// TrainTokenizer learns at most targetSize tokens from the supplied corpus.
// It supports configurable vocabulary sizes up to MaxVocabSize (32,768).
func TrainTokenizer(corpus []string, targetSize int) (*Tokenizer, error) {
	return TrainTokenizerWithOptions(corpus, TokenizerConfig{
		VocabSize: targetSize,
	})
}

// TrainTokenizerWithOptions trains a byte-level BPE tokenizer with specific configuration options,
// including special tokens and configurable target vocabulary sizes up to 32k.
func TrainTokenizerWithOptions(corpus []string, cfg TokenizerConfig) (*Tokenizer, error) {
	if len(corpus) == 0 {
		return nil, fmt.Errorf("tokenizer corpus is empty")
	}
	targetSize := cfg.VocabSize
	if targetSize < MinVocabSize {
		return nil, fmt.Errorf("tokenizer target size must be at least %d", MinVocabSize)
	}
	if targetSize > MaxVocabSize {
		return nil, fmt.Errorf("tokenizer target size must not exceed %d", MaxVocabSize)
	}

	t := &Tokenizer{
		Tokens:        make([]string, 0, targetSize),
		SpecialTokens: make(map[string]int),
		unkTokenID:    -1,
	}

	known := make(map[string]bool, targetSize)

	// 1. Initial 256 byte tokens: 0..255
	for i := 0; i < 256; i++ {
		bTok := string([]byte{byte(i)})
		t.Tokens = append(t.Tokens, bTok)
		known[bTok] = true
	}

	// 2. Register special tokens if provided
	if len(cfg.SpecialTokens) > 0 {
		// Sort special tokens for deterministic assignment
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

	// 3. Prepare sequences for BPE merge training.
	// Sequences are split so special tokens do not get merged with regular text.
	// Symbols are token ids; the initial 256 byte tokens hold ids 0..255, so a
	// raw byte is its own id until the first merge creates a new token.
	sequences := make([][]int32, 0, len(corpus))
	for _, text := range corpus {
		subTexts := t.splitSpecialTokens(text)
		for _, chunk := range subTexts {
			if chunk.isSpecial || len(chunk.text) == 0 {
				continue
			}
			seq := make([]int32, len(chunk.text))
			for i := 0; i < len(chunk.text); i++ {
				seq[i] = int32(chunk.text[i])
			}
			sequences = append(sequences, seq)
		}
	}

	// The merge loop resolves token strings to ids through the index, so build
	// it before learning merges (reindex also runs again on the final vocab).
	t.reindex()

	// 4. Learn BPE merges up to targetSize.
	t.trainMerges(sequences, targetSize)

	t.reindex()
	return t, nil
}

// pairItem is one merge candidate: an adjacent-pair key and its occurrence
// count. Keys are compared by count, then by the pair's token strings, which
// is the tie-break the previous full-sort implementation used.
type pairItem struct {
	count int
	key   uint64
}

// pairQueue is a max-heap of merge candidates. Counts change as merges are
// applied, so entries go stale; best discards or refreshes them on pop, and
// rebuild bounds the number of stale entries.
type pairQueue struct {
	tokens []string
	items  []pairItem
}

func (q *pairQueue) Len() int { return len(q.items) }

func (q *pairQueue) Less(i, j int) bool {
	a, b := q.items[i], q.items[j]
	if a.count != b.count {
		return a.count > b.count
	}
	leftA, rightA := q.tokens[uint32(a.key>>32)], q.tokens[uint32(a.key)]
	leftB, rightB := q.tokens[uint32(b.key>>32)], q.tokens[uint32(b.key)]
	if leftA != leftB {
		return leftA < leftB
	}
	return rightA < rightB
}

func (q *pairQueue) Swap(i, j int) { q.items[i], q.items[j] = q.items[j], q.items[i] }
func (q *pairQueue) Push(x any)    { q.items = append(q.items, x.(pairItem)) }

func (q *pairQueue) Pop() any {
	old := q.items
	n := len(old)
	item := old[n-1]
	q.items = old[:n-1]
	return item
}

// increment records one new occurrence of a pair.
func (q *pairQueue) increment(key uint64, counts map[uint64]int, blocked map[uint64]bool) {
	counts[key]++
	if counts[key] >= 2 && !blocked[key] {
		heap.Push(q, pairItem{count: counts[key], key: key})
	}
}

// best returns the merge to apply next: the highest count above one, ties
// broken by the smallest (left, right) token strings.
func (q *pairQueue) best(counts map[uint64]int, blocked map[uint64]bool) (pairItem, bool) {
	for q.Len() > 0 {
		item := heap.Pop(q).(pairItem)
		count := counts[item.key]
		if count != item.count {
			// Stale entry: the pair gained or lost occurrences since it was
			// pushed. Re-file it under its current count.
			if count >= 2 && !blocked[item.key] {
				heap.Push(q, pairItem{count: count, key: item.key})
			}
			continue
		}
		if count < 2 || blocked[item.key] {
			continue
		}
		return item, true
	}
	return pairItem{}, false
}

// rebuild drops stale entries so the queue stays proportional to live pairs.
func (q *pairQueue) rebuild(counts map[uint64]int, blocked map[uint64]bool) {
	q.items = q.items[:0]
	for key, count := range counts {
		if count >= 2 && !blocked[key] {
			q.items = append(q.items, pairItem{count: count, key: key})
		}
	}
	heap.Init(q)
}

// markBlockedSplits blocks every pair whose concatenation is already a token.
// A string can be tokenized more than one way ("ab"+"c" and "a"+"bc"), and
// the previous implementation never counted such a pair, so neither does this
// one. Splits of a newly created token are the only pairs that can become
// blocked later, because token ids are never removed.
func (t *Tokenizer) markBlockedSplits(token string, blocked map[uint64]bool) {
	for i := 1; i < len(token); i++ {
		left, leftOK := t.index[token[:i]]
		right, rightOK := t.index[token[i:]]
		if leftOK && rightOK {
			blocked[mergeKey(int32(left), int32(right))] = true
		}
	}
}

// trainMerges learns merge rules until the vocabulary reaches targetSize.
//
// Pair counts are seeded with one corpus scan and then adjusted only around
// the occurrences each merge replaces, so learning is linear in the corpus
// plus the work of the merges themselves instead of a full rescan (with a
// string-keyed map and a sort of every distinct pair) per merge. The merge
// chosen is identical to that rescan, including tie-breaking, so existing
// vocabularies and artifacts are unaffected.
func (t *Tokenizer) trainMerges(sequences [][]int32, targetSize int) {
	counts := make(map[uint64]int, 1<<14)
	blocked := make(map[uint64]bool)
	for _, token := range t.Tokens {
		t.markBlockedSplits(token, blocked)
	}

	// Seeding: every symbol is still a raw byte id (< 256), so the counts fit
	// in a dense table before the first merge.
	var seedCounts [1 << 16]int32
	for _, seq := range sequences {
		for i := 0; i+1 < len(seq); i++ {
			seedCounts[int(seq[i])<<8|int(seq[i+1])]++
		}
	}
	queue := &pairQueue{tokens: t.Tokens}
	for k, count := range seedCounts {
		if count < 2 {
			continue
		}
		key := mergeKey(int32(k>>8), int32(k&0xff))
		counts[key] = int(count)
		if !blocked[key] {
			queue.items = append(queue.items, pairItem{count: int(count), key: key})
		}
	}
	heap.Init(queue)

	for len(t.Tokens) < targetSize {
		item, ok := queue.best(counts, blocked)
		if !ok {
			break
		}
		left, right := int32(uint32(item.key>>32)), int32(uint32(item.key))
		merged := t.Tokens[left] + t.Tokens[right]
		t.Merges = append(t.Merges, Merge{Left: t.Tokens[left], Right: t.Tokens[right], Token: merged})
		t.Tokens = append(t.Tokens, merged)
		queue.tokens = t.Tokens // append may have moved the backing array
		id := int32(len(t.Tokens) - 1)
		t.index[merged] = int(id)
		t.markBlockedSplits(merged, blocked)
		t.applyMergeEverywhere(sequences, left, right, id, counts, blocked, queue)
		if queue.Len() > 4*len(counts)+1024 {
			queue.rebuild(counts, blocked)
		}
	}
}

// applyMergeEverywhere replaces every non-overlapping occurrence of the pair,
// scanning each sequence left to right like applyMerge, and keeps pair counts
// in step: a replacement at ... prev, left, right, next ... removes (prev,
// left), (left, right) and (right, next) and adds (prev, merged) and (merged,
// next). Pairs that the untouched remainder of the sequence forms are already
// counted, and (merged, next) is added here even when next is itself merged
// away later in this pass, which that merge then removes again.
func (t *Tokenizer) applyMergeEverywhere(sequences [][]int32, left, right, merged int32, counts map[uint64]int, blocked map[uint64]bool, queue *pairQueue) {
	for s := range sequences {
		seq := sequences[s]
		n := 0
		for i := 0; i < len(seq); {
			if i+1 < len(seq) && seq[i] == left && seq[i+1] == right {
				prev, next := int32(-1), int32(-1)
				if n > 0 {
					prev = seq[n-1]
				}
				if i+2 < len(seq) {
					next = seq[i+2]
				}
				if prev >= 0 {
					counts[mergeKey(prev, left)]--
				}
				counts[mergeKey(left, right)]--
				if next >= 0 {
					counts[mergeKey(right, next)]--
				}
				if prev >= 0 {
					queue.increment(mergeKey(prev, merged), counts, blocked)
				}
				if next >= 0 {
					queue.increment(mergeKey(merged, next), counts, blocked)
				}
				seq[n] = merged
				n++
				i += 2
				continue
			}
			seq[n] = seq[i]
			n++
			i++
		}
		sequences[s] = seq[:n]
	}
}

func byteSymbols(text string) []string {
	out := make([]string, len([]byte(text)))
	for i, b := range []byte(text) {
		out[i] = string([]byte{b})
	}
	return out
}

func applyMerge(seq []string, left, right, merged string) []string {
	out := make([]string, 0, len(seq))
	for i := 0; i < len(seq); {
		if i+1 < len(seq) && seq[i] == left && seq[i+1] == right {
			out = append(out, merged)
			i += 2
		} else {
			out = append(out, seq[i])
			i++
		}
	}
	return out
}

func (t *Tokenizer) indexOf(tok string) int {
	if t.index != nil {
		if id, ok := t.index[tok]; ok {
			return id
		}
	}
	for id, s := range t.Tokens {
		if s == tok {
			return id
		}
	}
	return -1
}

func (t *Tokenizer) reindex() {
	t.index = make(map[string]int, len(t.Tokens))
	for id, token := range t.Tokens {
		t.index[token] = id
	}
	t.mergeIndex = make(map[uint64]bpeRule, len(t.Merges))
	for rank, merge := range t.Merges {
		left, leftOK := t.index[merge.Left]
		right, rightOK := t.index[merge.Right]
		out, outOK := t.index[merge.Token]
		if !leftOK || !rightOK || !outOK {
			continue
		}
		key := mergeKey(int32(left), int32(right))
		if _, exists := t.mergeIndex[key]; exists {
			// Keep the earliest rule for a pair, matching replay order.
			continue
		}
		t.mergeIndex[key] = bpeRule{rank: uint32(rank), out: int32(out)}
	}
	t.hasByteIDs = true
	t.byteIDs = [256]int32{}
	for b := 0; b < 256; b++ {
		id, ok := t.index[string([]byte{byte(b)})]
		if !ok {
			t.hasByteIDs = false
			break
		}
		t.byteIDs[b] = int32(id)
	}
}

// SpecialTokenID returns the token ID associated with a special token, if present.
func (t *Tokenizer) SpecialTokenID(token string) (int, bool) {
	if t.SpecialTokens == nil {
		return 0, false
	}
	id, ok := t.SpecialTokens[token]
	return id, ok
}

// IsSpecialToken returns whether the given token string is registered as a special token.
func (t *Tokenizer) IsSpecialToken(token string) bool {
	if t.SpecialTokens == nil {
		return false
	}
	_, ok := t.SpecialTokens[token]
	return ok
}

// SetUnknownToken designates a special token as the fallback for unknown or out-of-vocab characters.
func (t *Tokenizer) SetUnknownToken(token string) error {
	id, ok := t.SpecialTokenID(token)
	if !ok {
		return fmt.Errorf("token %q is not a registered special token", token)
	}
	t.unkTokenID = id
	return nil
}

// UnknownTokenID returns the ID of the designated unknown token, or -1 if not set.
func (t *Tokenizer) UnknownTokenID() int {
	return t.unkTokenID
}

type textChunk struct {
	text      string
	isSpecial bool
	id        int
}

func (t *Tokenizer) splitSpecialTokens(text string) []textChunk {
	if len(t.SpecialTokens) == 0 {
		return []textChunk{{text: text, isSpecial: false}}
	}
	var res []textChunk
	curr := text
	for len(curr) > 0 {
		bestPos := -1
		bestLen := 0
		bestTok := ""
		bestID := -1
		for st, id := range t.SpecialTokens {
			pos := strings.Index(curr, st)
			if pos != -1 {
				if bestPos == -1 || pos < bestPos || (pos == bestPos && len(st) > bestLen) {
					bestPos = pos
					bestLen = len(st)
					bestTok = st
					bestID = id
				}
			}
		}
		if bestPos == -1 {
			res = append(res, textChunk{text: curr, isSpecial: false})
			break
		}
		if bestPos > 0 {
			res = append(res, textChunk{text: curr[:bestPos], isSpecial: false})
		}
		res = append(res, textChunk{text: bestTok, isSpecial: true, id: bestID})
		curr = curr[bestPos+bestLen:]
	}
	return res
}

// Encode turns text into token IDs by replaying learned BPE merges,
// preserving any registered special tokens.
func (t *Tokenizer) Encode(text string) []int {
	if t.index == nil {
		t.reindex()
	}

	chunks := t.splitSpecialTokens(text)
	var ids []int

	for _, chunk := range chunks {
		if chunk.isSpecial {
			ids = append(ids, chunk.id)
			continue
		}
		ids = append(ids, t.encodeChunk(chunk.text)...)
	}
	return ids
}

// encodeChunk tokenizes one non-special chunk.
func (t *Tokenizer) encodeChunk(text string) []int {
	if !t.hasByteIDs {
		// Tokenizer without the 256 byte tokens: the reference path is the
		// only correct one (it can also emit the unknown token).
		return t.encodeChunkSlow(text)
	}
	n := len(text)
	if n == 0 {
		return nil
	}
	// tokens holds each node's current token id; a merged-away node is
	// marked -1, which no rule can match because token ids are >= 0.
	tokens := make([]int32, n)
	prev := make([]int32, n)
	next := make([]int32, n)
	for i := 0; i < n; i++ {
		tokens[i] = t.byteIDs[text[i]]
		prev[i] = int32(i - 1)
		next[i] = int32(i + 1)
	}
	next[n-1] = -1
	queue := make(bpeQueue, 0, n)
	for i := 0; i+1 < n; i++ {
		if rule, ok := t.mergeIndex[mergeKey(tokens[i], tokens[i+1])]; ok {
			queue = append(queue, packPair(rule.rank, int32(i)))
		}
	}
	heap.Init(&queue)

	for queue.Len() > 0 {
		item := heap.Pop(&queue).(uint64)
		rank := uint32(item >> 32)
		pos := int32(uint32(item))
		right := next[pos]
		if tokens[pos] < 0 || right < 0 || tokens[right] < 0 {
			continue // node already merged away
		}
		rule, ok := t.mergeIndex[mergeKey(tokens[pos], tokens[right])]
		if !ok || rule.rank != rank {
			// Stale entry: a neighbouring merge changed this pair.
			continue
		}
		tokens[pos] = rule.out
		tokens[right] = -1
		next[pos] = next[right]
		if next[right] >= 0 {
			prev[next[right]] = pos
		}
		if right := next[pos]; right >= 0 {
			if rule, ok := t.mergeIndex[mergeKey(tokens[pos], tokens[right])]; ok {
				heap.Push(&queue, packPair(rule.rank, pos))
			}
		}
		if left := prev[pos]; left >= 0 {
			if rule, ok := t.mergeIndex[mergeKey(tokens[left], tokens[pos])]; ok {
				heap.Push(&queue, packPair(rule.rank, left))
			}
		}
	}

	ids := make([]int, 0, n)
	for i := int32(0); i >= 0; i = next[i] {
		ids = append(ids, int(tokens[i]))
	}
	return ids
}

// encodeChunkSlow is the original implementation: replay every merge in
// learned order over the whole symbol sequence. It is kept as the fallback
// for tokenizers without the 256 byte tokens, where byte ids cannot be
// mapped directly (and where unknown tokens may apply).
func (t *Tokenizer) encodeChunkSlow(text string) []int {
	seq := byteSymbols(text)
	for _, merge := range t.Merges {
		seq = applyMerge(seq, merge.Left, merge.Right, merge.Token)
	}
	ids := make([]int, 0, len(seq))
	for _, token := range seq {
		if id, ok := t.index[token]; ok {
			ids = append(ids, id)
		} else if t.unkTokenID >= 0 {
			ids = append(ids, t.unkTokenID)
		}
	}
	return ids
}

// Decode concatenates token bytes, omitting invalid IDs.
func (t *Tokenizer) Decode(ids []int) string {
	var out strings.Builder
	for _, id := range ids {
		if id >= 0 && id < len(t.Tokens) {
			out.WriteString(t.Tokens[id])
		}
	}
	return out.String()
}

// Save serializes the tokenizer, merges, special tokens, and metadata to a JSON artifact.
func (t *Tokenizer) Save(path string) error {
	tokensBytes := make([][]byte, len(t.Tokens))
	for i, tok := range t.Tokens {
		tokensBytes[i] = []byte(tok)
	}
	mergesBytes := make([]artifactMerge, len(t.Merges))
	for i, m := range t.Merges {
		mergesBytes[i] = artifactMerge{
			Left:  []byte(m.Left),
			Right: []byte(m.Right),
			Token: []byte(m.Token),
		}
	}

	artifact := TokenizerArtifact{
		Metadata: TokenizerMetadata{
			Version:   TokenizerArtifactVersion,
			VocabSize: len(t.Tokens),
			Config: TokenizerConfig{
				VocabSize:     len(t.Tokens),
				SpecialTokens: t.SpecialTokens,
			},
		},
		Tokens:        tokensBytes,
		Merges:        mergesBytes,
		SpecialTokens: t.SpecialTokens,
	}
	data, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return fmt.Errorf("encode tokenizer artifact: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// LoadTokenizer restores a Tokenizer from a saved JSON artifact.
func LoadTokenizer(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var artifact TokenizerArtifact
	if err := json.Unmarshal(data, &artifact); err != nil {
		return nil, fmt.Errorf("decode tokenizer artifact: %w", err)
	}

	toks := make([]string, len(artifact.Tokens))
	for i, b := range artifact.Tokens {
		toks[i] = string(b)
	}
	merges := make([]Merge, len(artifact.Merges))
	for i, m := range artifact.Merges {
		merges[i] = Merge{
			Left:  string(m.Left),
			Right: string(m.Right),
			Token: string(m.Token),
		}
	}

	tok := &Tokenizer{
		Tokens:        toks,
		Merges:        merges,
		SpecialTokens: artifact.SpecialTokens,
		unkTokenID:    -1,
	}
	if tok.SpecialTokens == nil {
		tok.SpecialTokens = make(map[string]int)
	}
	tok.reindex()
	return tok, nil
}
