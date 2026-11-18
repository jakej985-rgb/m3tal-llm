package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"m3tal-llm/internal/lm"
)

func main() {
	corpusPath := flag.String("corpus", "data/curriculum-train.txt", "path to training corpus file")
	chatPath := flag.String("chat", "data/chat-curriculum.jsonl", "path to chat curriculum file")
	vocabSize := flag.Int("vocab", 1536, "target vocabulary size")
	outPath := flag.String("out", "models/tokenizer-1536.json", "output path for tokenizer artifact")
	flag.Parse()

	if err := run(*corpusPath, *chatPath, *vocabSize, *outPath); err != nil {
		fmt.Fprintf(os.Stderr, "train-tokenizer: %v\n", err)
		os.Exit(1)
	}
}

func run(corpusPath, chatPath string, vocabSize int, outPath string) error {
	fmt.Printf("=== Phase 02: Training Calibrated Tokenizer (Vocab: %d) ===\n", vocabSize)

	// 1. Load training corpus texts
	var texts []string
	if corpusPath != "" {
		data, err := os.ReadFile(corpusPath)
		if err != nil {
			return fmt.Errorf("read corpus: %w", err)
		}
		// Split into distinct documents/paragraphs
		for _, doc := range strings.Split(string(data), "\n\n") {
			doc = strings.TrimSpace(doc)
			if len(doc) > 20 {
				texts = append(texts, doc)
			}
		}
		fmt.Printf("Loaded %d documents from %s\n", len(texts), corpusPath)
	}

	// 2. Load chat curriculum to ensure chat tokens and template boundaries are learned
	if chatPath != "" {
		f, err := os.Open(chatPath)
		if err == nil {
			chatCount := 0
			scanner := bufio.NewScanner(f)
			for scanner.Scan() {
				line := strings.TrimSpace(scanner.Text())
				if line == "" || strings.HasPrefix(line, "#") {
					continue
				}
				var ex struct {
					Messages []struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					} `json:"messages"`
				}
				if err := json.Unmarshal([]byte(line), &ex); err == nil {
					for _, m := range ex.Messages {
						texts = append(texts, m.Content)
					}
					chatCount++
				}
			}
			f.Close()
			fmt.Printf("Loaded %d chat messages from %s\n", chatCount*2, chatPath)
		}
	}

	if len(texts) == 0 {
		return fmt.Errorf("no training texts found")
	}

	// 3. Configure tokenizer with special tokens
	cfg := lm.TokenizerConfig{
		VocabSize: vocabSize,
		SpecialTokens: map[string]int{
			lm.DefaultEndOfTextToken: 256,
			lm.DefaultPadToken:       257,
			"<|im_start|>":           258,
			"<|im_end|>":             259,
		},
		UnknownToken: lm.DefaultEndOfTextToken,
	}

	fmt.Printf("Learning BPE merges up to %d tokens...\n", vocabSize)
	tok, err := lm.TrainTokenizerWithOptions(texts, cfg)
	if err != nil {
		return fmt.Errorf("train tokenizer: %w", err)
	}

	fmt.Printf("Tokenizer trained successfully: %d tokens, %d merges\n", len(tok.Tokens), len(tok.Merges))

	// 4. Verify core technical entities
	requiredEntities := []string{
		"M3TAL", "Core", "Hub", "plugin", "container", "docker",
		"Debian", "system", "health", "stats", "logs", "restart",
		"start", "stop", "Thought:", "PLAN:",
	}

	fmt.Printf("\nVerifying Semantic Token Coverage:\n")
	foundCount := 0
	for _, ent := range requiredEntities {
		encoded := tok.Encode(ent)
		encodedWithSpace := tok.Encode(" " + ent)
		if len(encoded) == 1 || len(encodedWithSpace) == 1 {
			foundCount++
			fmt.Printf("  ✓ %-12s (atomic single token)\n", ent)
		} else {
			fmt.Printf("  ~ %-12s (encodes in %d tokens)\n", ent, len(encoded))
		}
	}

	// 5. Measure compression ratio on curriculum text
	testSample := texts[0]
	if len(texts) > 5 {
		testSample = strings.Join(texts[:5], " ")
	}
	words := len(strings.Fields(testSample))
	encoded := tok.Encode(testSample)
	tokensPerWord := float64(len(encoded)) / float64(words)
	fmt.Printf("\nCompression Benchmark:\n")
	fmt.Printf("  Sample Words:       %d\n", words)
	fmt.Printf("  Encoded Tokens:     %d\n", len(encoded))
	fmt.Printf("  Tokens per Word:    %.2f (target <= 1.30)\n", tokensPerWord)

	// 6. Save verified tokenizer artifact
	if err := tok.Save(outPath); err != nil {
		return fmt.Errorf("save tokenizer: %w", err)
	}
	fp := lm.TokenizerFingerprint(tok)
	fmt.Printf("\nSaved Tokenizer Artifact:\n")
	fmt.Printf("  Path:        %s\n", outPath)
	fmt.Printf("  Fingerprint: %s\n", fp)

	return nil
}
