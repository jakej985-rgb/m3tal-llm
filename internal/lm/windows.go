package lm

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// This file implements conversation-aware window construction for chat
// SFT and DPO (issue #58: chat windows must be conversation-safe).
// Before this, both pipelines cropped the serialized token stream from
// the front, which could start a window in the middle of a message and
// leave the model predicting assistant content from a truncated user
// request.
//
// Truncation policy — one policy, used by SFT and DPO alike:
//
//  1. A message is atomic. Every window begins at a message header, so
//     no target token is ever predicted from half a message and no
//     assistant target is ever partially included.
//  2. The protected spine is: leading system messages, the last
//     non-assistant message (the active request), and every message
//     after it (the assistant replies). This is what the model is being
//     taught to answer.
//  3. If the protected spine does not fit the context window, the
//     example is dropped and counted with a reason. It is never silently
//     truncated.
//  4. Leftover budget is filled with whole earlier messages, newest
//     first, until the next message no longer fits. Older history is
//     dropped as whole messages and counted.
//  5. DPO computes one prompt for the pair, trimmed once against the
//     longer of the two replies, so chosen and rejected are always
//     scored under identical context. A pair whose replies leave no
//     usable prompt is dropped rather than compared asymmetrically.

// DropReason classifies why an example was dropped instead of trained
// on. Drops are data-policy outcomes, not bugs.
type DropReason string

const (
	// DropTooShort means the serialized sequence has fewer than two
	// tokens and cannot form an input/target pair.
	DropTooShort DropReason = "too_short"
	// DropNoAssistant means the window contains no assistant target.
	DropNoAssistant DropReason = "no_assistant_tokens"
	// DropOversizedSystem means the leading system messages alone do
	// not fit the context window.
	DropOversizedSystem DropReason = "oversized_system_prompt"
	// DropOversizedUser means the active request (with system messages)
	// does not fit the context window.
	DropOversizedUser DropReason = "oversized_user_request"
	// DropOversizedTarget means the request fits but the assistant
	// replies after it do not. For DPO it also covers a reply so long
	// that no shared chosen/rejected prompt fits around it.
	DropOversizedTarget DropReason = "oversized_assistant_target"
)

// windowDropError marks a build failure caused by the truncation policy
// rather than by a programming error.
type windowDropError struct {
	reason DropReason
	detail string
}

func (e *windowDropError) Error() string {
	if e.detail == "" {
		return string(e.reason)
	}
	return fmt.Sprintf("%s: %s", e.reason, e.detail)
}

func dropf(reason DropReason, format string, args ...any) error {
	return &windowDropError{reason: reason, detail: fmt.Sprintf(format, args...)}
}

// dropReasonOf reports the policy reason behind err, if any.
func dropReasonOf(err error) (DropReason, bool) {
	var de *windowDropError
	if errors.As(err, &de) {
		return de.reason, true
	}
	return "", false
}

// WindowReport records what a window build retained, trimmed, and
// dropped. It is the reporting surface for issue #58: every omitted
// example has a counted reason.
type WindowReport struct {
	Context         int              `json:"context"`
	Examples        int64            `json:"examples"`
	Retained        int64            `json:"retained"`
	Dropped         int64            `json:"dropped"`
	DroppedByReason map[string]int64 `json:"dropped_by_reason,omitempty"`
	ExamplesTrimmed int64            `json:"examples_trimmed"` // lost at least one whole message
	MessagesTrimmed int64            `json:"messages_trimmed"` // whole messages removed for budget
	WindowTokens    int64            `json:"window_tokens"`
	MaskedTokens    int64            `json:"masked_tokens"`
}

func newWindowReport(context int) WindowReport {
	return WindowReport{Context: context, DroppedByReason: map[string]int64{}}
}

func (r *WindowReport) retain(tokens, masked int64, trimmedMessages int) {
	r.Retained++
	r.WindowTokens += tokens
	r.MaskedTokens += masked
	if trimmedMessages > 0 {
		r.ExamplesTrimmed++
		r.MessagesTrimmed += int64(trimmedMessages)
	}
}

func (r *WindowReport) drop(err error) {
	reason, ok := dropReasonOf(err)
	if !ok {
		reason = "invalid"
	}
	r.Dropped++
	r.DroppedByReason[string(reason)]++
}

// DropSummary renders dropped-by-reason counts in stable order.
func (r *WindowReport) DropSummary() string { return FormatDropReasons(r.DroppedByReason) }

// FormatDropReasons renders a reason→count map in stable order.
func FormatDropReasons(counts map[string]int64) string {
	if len(counts) == 0 {
		return "none"
	}
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, counts[k]))
	}
	return strings.Join(parts, ", ")
}

// messageTokens is one message serialized to tokens with its loss mask.
// Messages are encoded separately so their boundaries are exact.
type messageTokens struct {
	role ChatRole
	ids  []int
	mask []bool
}

// messageTokens serializes every message independently, using the same
// segment encoding as Conversation.SerializedParts so window token
// streams match full-conversation serialization exactly.
func (c Conversation) messageTokens(t *Tokenizer) []messageTokens {
	out := make([]messageTokens, len(c.Messages))
	for i, m := range c.Messages {
		segs := []chatSegment{
			{imStart + string(m.Role) + "\n", false},
			{m.Content, m.Role == RoleAssistant},
			{imEnd + "\n", m.Role == RoleAssistant},
		}
		ids, mask := encodeSegments(t, segs)
		out[i] = messageTokens{role: m.Role, ids: ids, mask: mask}
	}
	return out
}

func sumTokens(blocks []messageTokens) int {
	total := 0
	for _, b := range blocks {
		total += len(b.ids)
	}
	return total
}

// selectMessages chooses whole message indices to keep within budget
// tokens. It returns the indices in order and the number of whole
// messages dropped for budget. An error is a policy drop, never a panic:
// callers count it and move on.
func selectMessages(blocks []messageTokens, budget int) ([]int, int, error) {
	n := len(blocks)
	if n == 0 {
		return nil, 0, dropf(DropTooShort, "conversation has no messages")
	}
	if budget < 2 {
		return nil, 0, dropf(DropTooShort, "context window holds %d tokens", budget)
	}
	// Leading system messages are protected.
	sysEnd := 0
	for sysEnd < n && blocks[sysEnd].role == RoleSystem {
		sysEnd++
	}
	// The active request is the last non-assistant message; everything
	// after it is the assistant replies being trained.
	req := -1
	for i := n - 1; i >= 0; i-- {
		if blocks[i].role != RoleAssistant {
			req = i
			break
		}
	}
	if req < 0 {
		return nil, 0, dropf(DropNoAssistant, "conversation has no request to answer")
	}
	// A request that is itself a leading system message folds into the
	// system prefix; nothing is duplicated.
	if sysEnd > req {
		sysEnd = req
	}

	systemTokens := sumTokens(blocks[:sysEnd])
	requestTokens := len(blocks[req].ids)
	tailTokens := sumTokens(blocks[req:])

	if systemTokens > budget {
		return nil, 0, dropf(DropOversizedSystem, "system prompt is %d tokens, window is %d", systemTokens, budget)
	}
	if systemTokens+requestTokens > budget {
		return nil, 0, dropf(DropOversizedUser, "request is %d tokens with %d system tokens, window is %d",
			requestTokens, systemTokens, budget)
	}
	if systemTokens+tailTokens > budget {
		return nil, 0, dropf(DropOversizedTarget, "assistant replies need %d tokens, only %d fit after %d system and %d request tokens",
			tailTokens-requestTokens, budget-systemTokens-requestTokens, systemTokens, requestTokens)
	}

	// Fill the remaining budget with whole earlier messages, newest
	// first, stopping at the first that does not fit so the kept history
	// stays contiguous.
	used := systemTokens + tailTokens
	start := req
	for i := req - 1; i >= sysEnd; i-- {
		if used+len(blocks[i].ids) > budget {
			break
		}
		used += len(blocks[i].ids)
		start = i
	}

	keep := make([]int, 0, sysEnd+(req-start)+(n-req))
	for i := 0; i < sysEnd; i++ {
		keep = append(keep, i)
	}
	for i := start; i < n; i++ {
		keep = append(keep, i)
	}
	return keep, start - sysEnd, nil
}

// assemble concatenates the kept message blocks.
func assemble(blocks []messageTokens, keep []int) (ids []int, mask []bool) {
	total := 0
	for _, i := range keep {
		total += len(blocks[i].ids)
	}
	ids = make([]int, 0, total)
	mask = make([]bool, 0, total)
	for _, i := range keep {
		ids = append(ids, blocks[i].ids...)
		mask = append(mask, blocks[i].mask...)
	}
	return ids, mask
}

// buildMaskedWindow converts a serialized sequence and its per-token
// mask into an input/target window. It rejects sequences that cannot
// form a training pair or that contain no assistant target.
func buildMaskedWindow(ids []int, idMask []bool) (sftWindow, error) {
	if len(ids) < 2 {
		return sftWindow{}, dropf(DropTooShort, "serialized sequence has %d tokens", len(ids))
	}
	if len(idMask) != len(ids) {
		return sftWindow{}, fmt.Errorf("mask length %d does not match %d tokens", len(idMask), len(ids))
	}
	w := sftWindow{
		input:   make([]int, len(ids)-1),
		targets: make([]int, len(ids)-1),
		mask:    make([]bool, len(ids)-1),
	}
	for i := range w.input {
		w.input[i] = ids[i]
		w.targets[i] = ids[i+1]
		w.mask[i] = idMask[i+1]
		if w.mask[i] {
			w.masked++
		}
	}
	if w.masked == 0 {
		return sftWindow{}, dropf(DropNoAssistant, "window contains no assistant target tokens")
	}
	return w, nil
}

// buildChatWindow applies the shared policy to a whole conversation.
func (m *Model) buildChatWindow(conv Conversation) (sftWindow, int, error) {
	blocks := conv.messageTokens(m.Tokenizer)
	keep, trimmed, err := selectMessages(blocks, m.Config.Context+1)
	if err != nil {
		return sftWindow{}, 0, err
	}
	ids, mask := assemble(blocks, keep)
	w, err := buildMaskedWindow(ids, mask)
	if err != nil {
		return sftWindow{}, 0, err
	}
	return w, trimmed, nil
}
