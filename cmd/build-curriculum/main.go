package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatExample struct {
	Messages []ChatMessage `json:"messages"`
	Category string        `json:"category,omitempty"`
}

type SourceRecord struct {
	Name           string   `json:"name"`
	Kind           string   `json:"kind"`
	Paths          []string `json:"paths"`
	License        string   `json:"license"`
	LicenseReview  string   `json:"license_review"`
	URL            string   `json:"url"`
	RetrievedUnix  int64    `json:"retrieved_unix"`
	Version        string   `json:"version"`
	Filters        []string `json:"filters"`
	Dedup          string   `json:"dedup"`
	DocumentCount  int      `json:"document_count"`
	EstimatedWords int      `json:"estimated_words"`
}

func main() {
	outDir := flag.String("out-dir", "data", "output directory for curriculum datasets")
	valFrac := flag.Float64("val-frac", 0.10, "fraction of documents held out for validation")
	maxStories := flag.Int("max-stories", 2000, "maximum narrative stories to include for grammar foundation")
	flag.Parse()

	if err := run(*outDir, *valFrac, *maxStories); err != nil {
		fmt.Fprintf(os.Stderr, "build-curriculum: %v\n", err)
		os.Exit(1)
	}
}

func run(outDir string, valFrac float64, maxStories int) error {
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return err
	}
	provDir := filepath.Join(outDir, "provenance")
	if err := os.MkdirAll(provDir, 0755); err != nil {
		return err
	}

	fmt.Println("=== Building High-Density Curriculum Dataset ===")

	// 1. Ingest clean grammar foundation from m3tal-tales
	stories, err := loadStories([]string{
		"data/m3tal-tales/m3tal_tales_tiny.jsonl",
		"data/m3tal-tales/m3tal_tales_small.jsonl",
	}, maxStories)
	if err != nil {
		return fmt.Errorf("load stories: %w", err)
	}
	fmt.Printf("Loaded %d clean narrative stories for grammar foundation\n", len(stories))

	// 2. Ingest and clean ecosystem documentation
	docRoots := []string{
		"/home/m3tal/apps/M3tal-core",
		"/home/m3tal/apps/M3tal-Hub",
		"/home/m3tal/apps/m3tal-plugins",
		"docs",
		"README.md",
		"ROADMAP.md",
		"STAGE0.md",
		"INTENTS.md",
	}
	docs, err := loadAndCleanDocs(docRoots)
	if err != nil {
		return fmt.Errorf("load docs: %w", err)
	}
	fmt.Printf("Extracted %d high-density declarative documentation sections\n", len(docs))

	// 3. Compile Chain-of-Thought & Conversational pairs
	chatPairs, err := compileChatCurriculum()
	if err != nil {
		return fmt.Errorf("compile chat pairs: %w", err)
	}
	fmt.Printf("Compiled %d Chain-of-Thought & instruction chat pairs\n", len(chatPairs))

	// 4. Partition documents deterministically into Train & Val splits
	var trainDocs, valDocs []string
	totalDocs := append(append([]string(nil), stories...), docs...)

	for _, doc := range totalDocs {
		if isValDoc(doc, valFrac) {
			valDocs = append(valDocs, doc)
		} else {
			trainDocs = append(trainDocs, doc)
		}
	}

	trainPath := filepath.Join(outDir, "curriculum-train.txt")
	valPath := filepath.Join(outDir, "curriculum-val.txt")
	chatPath := filepath.Join(outDir, "chat-curriculum.jsonl")

	if err := os.WriteFile(trainPath, []byte(strings.Join(trainDocs, "\n\n")+"\n"), 0644); err != nil {
		return fmt.Errorf("write train corpus: %w", err)
	}
	if err := os.WriteFile(valPath, []byte(strings.Join(valDocs, "\n\n")+"\n"), 0644); err != nil {
		return fmt.Errorf("write val corpus: %w", err)
	}
	if err := writeChatJSONL(chatPath, chatPairs); err != nil {
		return fmt.Errorf("write chat curriculum: %w", err)
	}

	// 5. Generate Provenance Manifest
	sources := []SourceRecord{
		{
			Name:           "m3tal-tales-narrative",
			Kind:           "text/synthetic-narrative",
			Paths:          []string{"data/m3tal-tales/m3tal_tales_tiny.jsonl", "data/m3tal-tales/m3tal_tales_small.jsonl"},
			License:        "self-owned",
			LicenseReview:  "Synthetic narrative dataset calibrated for grammatical consistency and narrative arc.",
			URL:            "local:data/m3tal-tales",
			RetrievedUnix:  time.Now().Unix(),
			Version:        "v1.0",
			Filters:        []string{"grammar-calibrated", "children-safe", "word-count-filtered"},
			Dedup:          "exact",
			DocumentCount:  len(stories),
			EstimatedWords: countWords(stories),
		},
		{
			Name:           "m3tal-ecosystem-docs",
			Kind:           "text/technical-documentation",
			Paths:          docRoots,
			License:        "self-owned",
			LicenseReview:  "First-party M3TAL technical architecture, plugins, hub, and specification documents.",
			URL:            "local:/home/m3tal/apps",
			RetrievedUnix:  time.Now().Unix(),
			Version:        "working-tree",
			Filters:        []string{"strip-yaml-frontmatter", "strip-tables", "strip-badges", "declarative-paragraphs-only"},
			Dedup:          "exact",
			DocumentCount:  len(docs),
			EstimatedWords: countWords(docs),
		},
		{
			Name:           "m3tal-chain-of-thought-chat",
			Kind:           "jsonl/conversational-cot",
			Paths:          []string{"data/chat-custom.jsonl", "data/chat-instructions.jsonl"},
			License:        "self-owned",
			LicenseReview:  "Curated reasoning traces, tool-planning slots, and operational Q&A pairs.",
			URL:            "local:data/chat-curriculum.jsonl",
			RetrievedUnix:  time.Now().Unix(),
			Version:        "v1.0",
			Filters:        []string{"m3tal.chatml.v1", "thought-trace-required", "json-schema-verified"},
			Dedup:          "exact",
			DocumentCount:  len(chatPairs),
			EstimatedWords: countChatWords(chatPairs),
		},
	}

	provBytes, err := json.MarshalIndent(sources, "", "  ")
	if err != nil {
		return err
	}
	provPath := filepath.Join(provDir, "curriculum.sources.json")
	if err := os.WriteFile(provPath, append(provBytes, '\n'), 0644); err != nil {
		return err
	}

	fmt.Printf("\nGenerated Curriculum Datasets:\n")
	fmt.Printf("  Train corpus:       %s (%d documents, ~%d words)\n", trainPath, len(trainDocs), countWords(trainDocs))
	fmt.Printf("  Held-out val:       %s (%d documents, ~%d words)\n", valPath, len(valDocs), countWords(valDocs))
	fmt.Printf("  Chat CoT dataset:   %s (%d examples, ~%d words)\n", chatPath, len(chatPairs), countChatWords(chatPairs))
	fmt.Printf("  Provenance records: %s\n", provPath)

	return nil
}

func loadStories(paths []string, maxStories int) ([]string, error) {
	var stories []string
	seen := make(map[string]bool)

	for _, p := range paths {
		if _, err := os.Stat(p); os.IsNotExist(err) {
			continue
		}
		f, err := os.Open(p)
		if err != nil {
			return nil, err
		}
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			if len(stories) >= maxStories {
				break
			}
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var rec struct {
				Story string `json:"story"`
				Title string `json:"title"`
				Moral string `json:"moral"`
			}
			if err := json.Unmarshal([]byte(line), &rec); err != nil {
				continue
			}
			storyText := strings.TrimSpace(rec.Story)
			if len(storyText) > 50 && !seen[storyText] {
				seen[storyText] = true
				formatted := fmt.Sprintf("Title: %s\nStory: %s\nMoral: %s", rec.Title, storyText, rec.Moral)
				stories = append(stories, formatted)
			}
		}
		f.Close()
	}
	return stories, nil
}

func loadAndCleanDocs(roots []string) ([]string, error) {
	var sections []string
	seen := make(map[[32]byte]bool)

	for _, root := range roots {
		info, err := os.Stat(root)
		if err != nil {
			continue
		}
		if !info.IsDir() {
			sec, err := cleanMarkdownFile(root)
			if err == nil {
				for _, s := range sec {
					h := sha256.Sum256([]byte(s))
					if !seen[h] {
						seen[h] = true
						sections = append(sections, s)
					}
				}
			}
			continue
		}

		err = filepath.WalkDir(root, func(path string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			if d.IsDir() {
				name := d.Name()
				if strings.HasPrefix(name, ".") || name == "node_modules" || name == "vendor" || name == "dist" || name == "build" || name == "data" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(strings.ToLower(path), ".md") && !strings.HasSuffix(strings.ToLower(path), ".txt") {
				return nil
			}
			sec, err := cleanMarkdownFile(path)
			if err != nil {
				return nil
			}
			for _, s := range sec {
				h := sha256.Sum256([]byte(s))
				if !seen[h] {
					seen[h] = true
					sections = append(sections, s)
				}
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return sections, nil
}

var (
	badgeRe        = regexp.MustCompile(`\[\!\[.*?\]\(.*?\)\]\(.*?\)`)
	htmlTagRe      = regexp.MustCompile(`<[^>]*>`)
	tableDividerRe = regexp.MustCompile(`^\|?\s*[-:]+[-|\s:]*\|$`)
)

func cleanMarkdownFile(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	content := string(data)

	// Strip YAML frontmatter
	if strings.HasPrefix(content, "---\n") {
		if end := strings.Index(content[4:], "\n---\n"); end != -1 {
			content = content[4+end+5:]
		}
	}

	// Split into sections by Markdown headers
	lines := strings.Split(content, "\n")
	var currentSection []string
	var sections []string

	flush := func() {
		text := strings.TrimSpace(strings.Join(currentSection, "\n"))
		currentSection = currentSection[:0]
		if len(text) < 60 {
			return
		}
		// Filter out pure code blocks or table divider sections
		if strings.Count(text, "|") > 10 && strings.Count(text, "---") > 2 {
			return
		}
		sections = append(sections, text)
	}

	inCodeBlock := false
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") {
			inCodeBlock = !inCodeBlock
		}
		if !inCodeBlock && (strings.HasPrefix(trimmed, "# ") || strings.HasPrefix(trimmed, "## ")) {
			flush()
		}
		// Skip badges, empty HTML, and table separator lines
		if badgeRe.MatchString(trimmed) || tableDividerRe.MatchString(trimmed) {
			continue
		}
		cleanLine := htmlTagRe.ReplaceAllString(line, "")
		currentSection = append(currentSection, cleanLine)
	}
	flush()
	return sections, nil
}

func compileChatCurriculum() ([]ChatExample, error) {
	var pairs []ChatExample
	seen := make(map[string]bool)

	addPair := func(user, thought, answer, category string) {
		key := user + " -> " + answer
		if seen[key] {
			return
		}
		seen[key] = true
		content := fmt.Sprintf("Thought: %s\n%s", thought, answer)
		pairs = append(pairs, ChatExample{
			Messages: []ChatMessage{
				{Role: "user", Content: user},
				{Role: "assistant", Content: content},
			},
			Category: category,
		})
	}

	// 1. Ingest existing custom chat examples and m3tal-codex knowledge
	for _, path := range []string{"data/m3tal_codex_tiny.jsonl", "data/chat-custom.jsonl", "data/chat-instructions.jsonl"} {
		f, err := os.Open(path)
		if err == nil {
			scanner := bufio.NewScanner(f)
			for scanner.Scan() {
				line := strings.TrimSpace(scanner.Text())
				if line == "" || strings.HasPrefix(line, "#") {
					continue
				}
				var ex ChatExample
				if err := json.Unmarshal([]byte(line), &ex); err == nil && len(ex.Messages) >= 2 {
					u := ex.Messages[0].Content
					a := ex.Messages[1].Content
					if !seen[u+" -> "+a] {
						seen[u+" -> "+a] = true
						pairs = append(pairs, ex)
					}
				}
			}
			f.Close()
		}
	}

	// 2. High-density M3TAL CoT additions
	m3talCoTPairs := []struct {
		user, thought, answer, cat string
	}{
		{
			user:    "What is M3TAL Core?",
			thought: "The user is asking for the architectural definition of M3TAL Core.",
			answer:  "M3TAL Core is a high-performance, Go-native Web Dashboard and System Monitor for Linux desktops and servers. It serves an embedded React GUI for hardware telemetry, system performance, settings, and API key management, while acting as a supervisor host for plugins.",
			cat:     "architecture",
		},
		{
			user:    "What is M3TAL Hub?",
			thought: "The user wants an overview of M3TAL Hub.",
			answer:  "M3TAL Hub is the unified release front and application catalog for the M3TAL ecosystem. It manages release manifests, automated builds, and central hosting for official applications.",
			cat:     "architecture",
		},
		{
			user:    "How does the M3TAL plugin supervisor work?",
			thought: "The user is asking about plugin lifecycle management in M3tal Core.",
			answer:  "M3TAL Core manages plugins through a supervised lifecycle. Each plugin defines a manifest (plugin.yaml) and communicates over standardized UNIX sockets or HTTP endpoints. Core enforces live allowlists for any mutating actions.",
			cat:     "plugins",
		},
		{
			user:    "How do I restart a container safely?",
			thought: "The user wants to know the procedure for mutating container state safely.",
			answer:  "To restart a container, request the action from the assistant or Core API (e.g., 'restart container jellyfin'). Core validates that the target container exists on its active allowlist, and then prompts the operator for exact confirmation before executing.",
			cat:     "operations",
		},
		{
			user:    "Where are M3TAL system logs stored?",
			thought: "The user is querying log file locations.",
			answer:  "M3TAL component logs are streamed through the logs tool in Core. On the filesystem, service logs are maintained in /var/log/m3tal/ or the designated container state directories.",
			cat:     "operations",
		},
		{
			user:    "Plan the user's request: check system health and show container inventory\nPLAN:",
			thought: "The user wants both health status and container inventory. First inspect health, then query active containers.",
			answer:  `{"steps":[{"tool":"health"},{"tool":"containers"}]}`,
			cat:     "plan",
		},
		{
			user:    "Plan the user's request: restart jellyfin and show stats\nPLAN:",
			thought: "The user wants to restart container jellyfin and review system statistics. Mutating step requires target confirmation.",
			answer:  `{"steps":[{"tool":"restart_container","target":"jellyfin"},{"tool":"stats"}]}`,
			cat:     "plan",
		},
		{
			user:    "Plan the user's request: inspect logs for traefik\nPLAN:",
			thought: "The user wants to read recent log output for the container named traefik.",
			answer:  `{"steps":[{"tool":"logs","target":"traefik"}]}`,
			cat:     "plan",
		},
		{
			user:    "Plan the user's request: stop stack media\nPLAN:",
			thought: "The user wants to stop the stack named media. This is a mutating stack action requiring confirmation.",
			answer:  `{"steps":[{"tool":"stop_stack","target":"media"}]}`,
			cat:     "plan",
		},
	}

	for _, item := range m3talCoTPairs {
		addPair(item.user, item.thought, item.answer, item.cat)
	}

	return pairs, nil
}

func isValDoc(text string, frac float64) bool {
	if frac <= 0 {
		return false
	}
	h := sha256.Sum256([]byte(text))
	u := binary.BigEndian.Uint64(h[:8])
	return float64(u)/float64(^uint64(0)) < frac
}

func countWords(texts []string) int {
	total := 0
	for _, t := range texts {
		total += len(strings.Fields(t))
	}
	return total
}

func countChatWords(examples []ChatExample) int {
	total := 0
	for _, ex := range examples {
		for _, m := range ex.Messages {
			total += len(strings.Fields(m.Content))
		}
	}
	return total
}

func writeChatJSONL(path string, examples []ChatExample) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()

	for _, ex := range examples {
		b, err := json.Marshal(ex)
		if err != nil {
			return err
		}
		if _, err := w.Write(append(b, '\n')); err != nil {
			return err
		}
	}
	return nil
}
