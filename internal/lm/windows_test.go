package lm

import (
	"fmt"
	"strings"
	"testing"
)

// Tests for conversation-safe window construction (issue #58). The
// acceptance criteria are: whole-message windows only, protected system
// instructions and active request, complete assistant targets, one
// shared DPO prompt, and counted retained/dropped/trimmed statistics.

func windowTestModel(t *testing.T, context int) *Model {
	t.Helper()
	tok := testChatTokenizer(t)
	m, err := NewModel(Config{Context: context, Width: 16, Layers: 1, Heads: 2}, tok, 1)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// windowIDs reconstructs the serialized ids of a window: input plus the
// final target.
func windowIDs(w sftWindow) []int {
	ids := append([]int(nil), w.input...)
	if len(w.targets) > 0 {
		ids = append(ids, w.targets[len(w.targets)-1])
	}
	return ids
}

// longHistoryConversation builds a system prompt, several complete turns,
// and a final answered request.
func longHistoryConversation(turns int) Conversation {
	msgs := []ChatMessage{{Role: RoleSystem, Content: "You are helpful."}}
	for i := 0; i < turns; i++ {
		msgs = append(msgs,
			ChatMessage{Role: RoleUser, Content: fmt.Sprintf("Hello there number %d.", i)},
			ChatMessage{Role: RoleAssistant, Content: "Hi! How can I help?"},
		)
	}
	msgs = append(msgs,
		ChatMessage{Role: RoleUser, Content: "What is 2+2?"},
		ChatMessage{Role: RoleAssistant, Content: "4"},
	)
	return Conversation{Messages: msgs}
}

func TestSFTWindowWholeMessagesOnly(t *testing.T) {
	conv := longHistoryConversation(6)
	tok := testChatTokenizer(t)
	full := sumTokens(conv.messageTokens(tok))
	// Leave room for the protected spine but not for all history.
	m := windowTestModel(t, full-10)
	w, trimmed, err := m.buildChatWindow(conv)
	if err != nil {
		t.Fatal(err)
	}
	if trimmed == 0 {
		t.Fatalf("expected history trimming with context %d (full sequence %d tokens)", m.Config.Context, full)
	}
	if len(w.input)+1 > m.Config.Context+1 {
		t.Fatalf("window has %d tokens; context is %d", len(w.input)+1, m.Config.Context)
	}
	decoded := m.Tokenizer.Decode(windowIDs(w))
	if !strings.HasPrefix(decoded, imStart+"system\n") {
		t.Fatalf("window does not start at the system header: %q", decoded[:min(len(decoded), 40)])
	}
	if !strings.HasSuffix(decoded, imEnd+"\n") {
		t.Fatalf("window does not end at a message boundary: %q", decoded)
	}
	if got, want := strings.Count(decoded, imStart), strings.Count(decoded, imEnd); got != want {
		t.Fatalf("window contains %d message starts and %d ends (partial message): %q", got, want, decoded)
	}
	if !strings.Contains(decoded, "You are helpful.") {
		t.Fatal("system instructions were dropped")
	}
	if !strings.Contains(decoded, "What is 2+2?") || !strings.Contains(decoded, imStart+"assistant\n4"+imEnd) {
		t.Fatalf("active request or its reply was dropped: %q", decoded)
	}
}

func TestSFTWindowAllFitMatchesSerialization(t *testing.T) {
	conv := longHistoryConversation(1)
	m := windowTestModel(t, 256)
	wantIDs, wantMask := conv.SerializedParts(m.Tokenizer)
	w, trimmed, err := m.buildChatWindow(conv)
	if err != nil {
		t.Fatal(err)
	}
	if trimmed != 0 {
		t.Fatalf("trimmed %d messages although the conversation fits", trimmed)
	}
	got := windowIDs(w)
	if len(got) != len(wantIDs) {
		t.Fatalf("window has %d ids, serialization has %d", len(got), len(wantIDs))
	}
	for i := range got {
		if got[i] != wantIDs[i] {
			t.Fatalf("id %d differs: window %d, serialization %d", i, got[i], wantIDs[i])
		}
	}
	for i := range w.mask {
		if w.mask[i] != wantMask[i+1] {
			t.Fatalf("mask %d differs from per-token serialization mask", i)
		}
	}
}

func TestSFTWindowDropsOversizedSystem(t *testing.T) {
	m := windowTestModel(t, 8)
	conv := Conversation{Messages: []ChatMessage{
		{Role: RoleSystem, Content: strings.Repeat("You are helpful. ", 20)},
		{Role: RoleUser, Content: "Hi"},
		{Role: RoleAssistant, Content: "Hello!"},
	}}
	_, report, err := m.PrepareChatExamples([]ChatExample{{Conversation: conv}})
	if err == nil {
		t.Fatal("oversized system prompt was accepted")
	}
	if report.Dropped != 1 || report.DroppedByReason[string(DropOversizedSystem)] != 1 {
		t.Fatalf("dropped=%d reasons=%s, want one %s", report.Dropped, report.DropSummary(), DropOversizedSystem)
	}
}

func TestSFTWindowDropsOversizedUserRequest(t *testing.T) {
	m := windowTestModel(t, 8)
	conv := Conversation{Messages: []ChatMessage{
		{Role: RoleUser, Content: strings.Repeat("What is 2+2? ", 20)},
		{Role: RoleAssistant, Content: "4"},
	}}
	_, report, err := m.PrepareChatExamples([]ChatExample{{Conversation: conv}})
	if err == nil {
		t.Fatal("oversized request was accepted")
	}
	if report.DroppedByReason[string(DropOversizedUser)] != 1 {
		t.Fatalf("reasons=%s, want one %s", report.DropSummary(), DropOversizedUser)
	}
}

func TestSFTWindowDropsOversizedAssistantTarget(t *testing.T) {
	// Budget must hold the "Hi" request but not request plus reply.
	m := windowTestModel(t, 40)
	conv := Conversation{Messages: []ChatMessage{
		{Role: RoleUser, Content: "Hi"},
		{Role: RoleAssistant, Content: strings.Repeat("Hi! How can I help? ", 20)},
	}}
	_, report, err := m.PrepareChatExamples([]ChatExample{{Conversation: conv}})
	if err == nil {
		t.Fatal("oversized assistant target was accepted")
	}
	if report.DroppedByReason[string(DropOversizedTarget)] != 1 {
		t.Fatalf("reasons=%s, want one %s", report.DropSummary(), DropOversizedTarget)
	}
}

func TestSFTWindowNeverStartsWithAssistantTail(t *testing.T) {
	// A window that only had room for trailing assistant tokens would
	// train the model to predict replies with no request. It must be
	// dropped instead.
	m := windowTestModel(t, 12)
	conv := Conversation{Messages: []ChatMessage{
		{Role: RoleUser, Content: strings.Repeat("Hello there. ", 30)},
		{Role: RoleAssistant, Content: "Hi! How can I help?"},
	}}
	windows, report, err := m.PrepareChatExamples([]ChatExample{{Conversation: conv}})
	if err == nil {
		for _, w := range windows {
			if w.mask[0] {
				t.Fatal("window begins with an assistant target and no request")
			}
		}
	}
	if report.Retained != 0 {
		t.Fatalf("retained %d windows for an unanswerable conversation", report.Retained)
	}
}

func TestSFTWindowReportCounts(t *testing.T) {
	m := windowTestModel(t, 256)
	examples := append(sftTestExamples(),
		ChatExample{Conversation: Conversation{Messages: []ChatMessage{
			{Role: RoleSystem, Content: strings.Repeat("You are helpful. ", 40)},
			{Role: RoleUser, Content: "Hi"},
			{Role: RoleAssistant, Content: "Hello!"},
		}}},
	)
	windows, report, err := m.PrepareChatExamples(examples)
	if err != nil {
		t.Fatal(err)
	}
	if report.Examples != int64(len(examples)) {
		t.Fatalf("examples=%d want %d", report.Examples, len(examples))
	}
	if report.Retained+report.Dropped != report.Examples {
		t.Fatalf("retained %d + dropped %d != examples %d", report.Retained, report.Dropped, report.Examples)
	}
	if int64(len(windows)) != report.Retained {
		t.Fatalf("%d windows but report says %d retained", len(windows), report.Retained)
	}
	if report.Dropped != 1 {
		t.Fatalf("dropped=%d want 1", report.Dropped)
	}
	if report.MaskedTokens <= 0 || report.WindowTokens <= 0 {
		t.Fatalf("token counts not recorded: %+v", report)
	}
}

// dpoPromptLen returns how many leading window tokens are shared prompt.
func dpoPromptLen(t *testing.T, m *Model, reply string, w dpoWindow) int {
	t.Helper()
	blocks := Conversation{Messages: []ChatMessage{{Role: RoleAssistant, Content: reply}}}.messageTokens(m.Tokenizer)
	return len(w.input) + 1 - len(blocks[0].ids)
}

func TestDPOWindowsShareIdenticalPrompt(t *testing.T) {
	m := windowTestModel(t, 256)
	pair := PreferencePair{
		ID: "shared",
		Prompt: []ChatMessage{
			{Role: RoleSystem, Content: "You are helpful."},
			{Role: RoleUser, Content: "What is 2+2?"},
		},
		Chosen:   "The answer is 4.",
		Rejected: "I don't know.",
	}
	cw, rw, _, err := m.dpoWindows(pair)
	if err != nil {
		t.Fatal(err)
	}
	pc := dpoPromptLen(t, m, pair.Chosen, cw)
	pr := dpoPromptLen(t, m, pair.Rejected, rw)
	if pc != pr {
		t.Fatalf("chosen prompt is %d tokens, rejected prompt is %d", pc, pr)
	}
	if pc <= 0 {
		t.Fatalf("prompt length %d", pc)
	}
	for i := 0; i < pc; i++ {
		if cw.input[i] != rw.input[i] {
			t.Fatalf("prompt token %d differs between chosen and rejected (%d vs %d)", i, cw.input[i], rw.input[i])
		}
	}
	if cw.active == 0 || rw.active == 0 {
		t.Fatal("a side has no assistant targets")
	}
}

func TestDPOWindowsAsymmetricRepliesShareTrimmedPrompt(t *testing.T) {
	m := windowTestModel(t, 256)
	// Many prompt turns so the shared prompt must be trimmed, and very
	// different reply lengths.
	prompt := []ChatMessage{{Role: RoleSystem, Content: "You are helpful."}}
	for i := 0; i < 12; i++ {
		prompt = append(prompt, ChatMessage{Role: RoleUser, Content: fmt.Sprintf("Hello there number %d.", i)})
	}
	pair := PreferencePair{
		ID:       "asymmetric",
		Prompt:   prompt,
		Chosen:   strings.Repeat("Hi! How can I help? ", 4),
		Rejected: "No.",
	}
	cw, rw, trimmed, err := m.dpoWindows(pair)
	if err != nil {
		t.Fatal(err)
	}
	if trimmed == 0 {
		t.Fatal("expected the shared prompt to drop whole history messages")
	}
	if len(cw.input)+1 > m.Config.Context+1 || len(rw.input)+1 > m.Config.Context+1 {
		t.Fatalf("window exceeds context: chosen %d, rejected %d, context %d",
			len(cw.input)+1, len(rw.input)+1, m.Config.Context)
	}
	pc := dpoPromptLen(t, m, pair.Chosen, cw)
	pr := dpoPromptLen(t, m, pair.Rejected, rw)
	if pc != pr {
		t.Fatalf("asymmetric prompts: chosen %d, rejected %d", pc, pr)
	}
	for i := 0; i < pc; i++ {
		if cw.input[i] != rw.input[i] {
			t.Fatalf("prompt token %d differs between chosen and rejected", i)
		}
	}
}

func TestDPOWindowsDropsWhenNoSharedPromptFits(t *testing.T) {
	m := windowTestModel(t, 8)
	pair := PreferencePair{
		ID:       "too-long",
		Prompt:   []ChatMessage{{Role: RoleUser, Content: "Hi"}},
		Chosen:   strings.Repeat("Hi! How can I help? ", 20),
		Rejected: "No.",
	}
	_, _, _, err := m.dpoWindows(pair)
	reason, ok := dropReasonOf(err)
	if !ok {
		t.Fatalf("error is not a policy drop: %v", err)
	}
	if reason != DropOversizedTarget {
		t.Fatalf("reason=%s, want %s", reason, DropOversizedTarget)
	}
}

func TestTrainDPOWithReportCountsDrops(t *testing.T) {
	m, ref := dpoModels(t)
	pairs := []PreferencePair{
		dpoTestPair(),
		{
			ID:       "oversized",
			Prompt:   []ChatMessage{{Role: RoleUser, Content: "Hi"}},
			Chosen:   strings.Repeat("Hi! How can I help? ", 40),
			Rejected: "No.",
		},
	}
	cfg := DPOConfig{TrainConfig: TrainConfig{MaxSteps: 2, LearningRate: 0.01, Seed: 1}, Beta: 0.1}
	_, report, err := m.TrainDPOWithReport(pairs, ref, cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	if report.Examples != 2 || report.Retained != 1 || report.Dropped != 1 {
		t.Fatalf("report: examples=%d retained=%d dropped=%d (%s)",
			report.Examples, report.Retained, report.Dropped, report.DropSummary())
	}
	if report.DroppedByReason[string(DropOversizedTarget)] != 1 {
		t.Fatalf("unexpected drop reasons: %s", report.DropSummary())
	}
}

func TestSelectMessagesBoundaries(t *testing.T) {
	m := windowTestModel(t, 128)
	cases := []struct {
		name  string
		msgs  []ChatMessage
		drop  DropReason
		check func(t *testing.T, w sftWindow, trimmed int)
	}{
		{
			name: "single-turn fits",
			msgs: []ChatMessage{{Role: RoleUser, Content: "Hi"}, {Role: RoleAssistant, Content: "Hello!"}},
			check: func(t *testing.T, w sftWindow, trimmed int) {
				if trimmed != 0 {
					t.Fatalf("trimmed %d messages", trimmed)
				}
			},
		},
		{
			// A trailing unanswered turn is legal: the earlier assistant
			// reply is still a complete target with its request in-window.
			name: "trailing unanswered turn",
			msgs: []ChatMessage{{Role: RoleUser, Content: "Hi"}, {Role: RoleAssistant, Content: "Hello!"}, {Role: RoleUser, Content: "Bye"}},
			check: func(t *testing.T, w sftWindow, trimmed int) {
				if trimmed != 0 {
					t.Fatalf("trimmed %d messages", trimmed)
				}
				if w.masked == 0 {
					t.Fatal("assistant reply lost its loss mask")
				}
			},
		},
		{
			name: "system only",
			msgs: []ChatMessage{{Role: RoleSystem, Content: "You are helpful."}},
			drop: DropNoAssistant,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w, trimmed, err := m.buildChatWindow(Conversation{Messages: tc.msgs})
			if tc.drop != "" {
				reason, ok := dropReasonOf(err)
				if !ok || reason != tc.drop {
					t.Fatalf("err=%v reason=%s, want %s", err, reason, tc.drop)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			tc.check(t, w, trimmed)
		})
	}
}
