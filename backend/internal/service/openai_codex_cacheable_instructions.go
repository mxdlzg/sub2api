package service

import (
	"fmt"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func codexDeveloperMessage(instructions string) map[string]any {
	return map[string]any{
		"type": "message",
		"role": "developer",
		"content": []any{
			map[string]any{"type": "input_text", "text": instructions},
		},
	}
}

// Some Codex versions sent their base prompt in instructions. Match the
// shipped template, including its interpolated IDs, before relocating it.
// Keep any trailing turn instructions in their original request field.
func splitLegacyCodexBaseInstructions(instructions, base string) (string, string) {
	remaining := instructions
	for {
		literal, variable, hasVariable := strings.Cut(base, "{{")
		var matches bool
		remaining, matches = strings.CutPrefix(remaining, literal)
		if !matches {
			return "", instructions
		}
		if !hasVariable {
			if turnInstructions, ok := strings.CutPrefix(remaining, "\n\n"); ok {
				return instructions[:len(instructions)-len(remaining)], turnInstructions
			}
			if strings.TrimSpace(remaining) == "" {
				return instructions[:len(instructions)-len(remaining)], ""
			}
			return "", instructions
		}
		name, rest, _ := strings.Cut(variable, "}}")
		if name != "connector_id" && name != "plugin_id" {
			return "", instructions
		}
		nextLiteral, _, _ := strings.Cut(rest, "{{")
		index := strings.Index(remaining, nextLiteral)
		if index < 0 || index > 128 || strings.ContainsAny(remaining[:index], "\r\n") {
			return "", instructions
		}
		remaining, base = remaining[index:], rest
	}
}

// GPT-6.2 bills top-level instructions as uncached input. Preserve replayed
// messages in order and prepend only instructions known to be reusable.
func normalizeCodexCacheableInstructions(reqBody map[string]any, model string, synthesize, compact bool) bool {
	input, _ := reqBody["input"].([]any)
	continuation, _ := reqBody["previous_response_id"].(string)
	canPrepend := !compact && strings.TrimSpace(continuation) == ""
	hasInstructions := false
	modified := false
	for _, item := range input {
		message, ok := item.(map[string]any)
		if !ok {
			continue
		}
		switch message["type"] {
		case "compaction", "compaction_summary":
			canPrepend = false
		}
		switch message["role"] {
		case "system":
			message["role"] = "developer"
			modified = true
			hasInstructions = true
		case "developer":
			hasInstructions = true
		}
	}
	if compact {
		if _, exists := reqBody["instructions"]; exists {
			delete(reqBody, "instructions")
			return true
		}
		return modified
	}

	instructions, _ := reqBody["instructions"].(string)
	var stable string
	if canPrepend {
		if strings.TrimSpace(instructions) != "" {
			stable, instructions = splitLegacyCodexBaseInstructions(instructions, defaultCodexSynthInstructions(model))
		} else if synthesize && !hasInstructions {
			stable = defaultCodexSynthInstructions(model)
		}
	}
	if stable != "" {
		firstText, _ := leadingCodexDeveloperInstructions(input)
		if firstText != stable {
			switch original := reqBody["input"].(type) {
			case nil, []any:
			case string:
				if strings.TrimSpace(original) != "" {
					input = []any{map[string]any{"type": "message", "role": "user", "content": original}}
				}
			default:
				return modified
			}
			reqBody["input"] = append([]any{codexDeveloperMessage(stable)}, input...)
		}
		reqBody["instructions"] = instructions
		return true
	}
	return modified
}

func leadingCodexDeveloperInstructions(input []any) (string, bool) {
	if len(input) == 0 {
		return "", false
	}
	message, ok := input[0].(map[string]any)
	if !ok || message["role"] != "developer" {
		return "", false
	}
	text, lossless := extractLosslessTextFromContent(message["content"])
	return strings.TrimSpace(text), lossless
}

// Raw forwarding paths keep the rest of the payload in its original wire
// shape. Decode just input when a request may need an instruction rewrite.
func normalizeCodexCacheableInstructionsBody(body []byte, model string, synthesize, compact bool) ([]byte, error) {
	if openai.GPT62ModelID(model) == "" {
		return body, nil
	}
	fields := gjson.GetManyBytes(body, "input", "instructions", "previous_response_id")
	input, instructions, continuation := fields[0], fields[1], fields[2]
	hasSystem, hasDeveloper := false, false
	if input.IsArray() {
		input.ForEach(func(_, item gjson.Result) bool {
			switch item.Get("role").String() {
			case "system":
				hasSystem = true
			case "developer":
				hasDeveloper = true
			}
			return !hasSystem
		})
	}
	if !hasSystem {
		switch {
		case compact:
			if !instructions.Exists() {
				return body, nil
			}
			return sjson.DeleteBytes(body, "instructions")
		case strings.TrimSpace(continuation.String()) != "":
			return body, nil
		case strings.TrimSpace(instructions.String()) != "":
			if stable, _ := splitLegacyCodexBaseInstructions(instructions.String(), defaultCodexSynthInstructions(model)); stable == "" {
				return body, nil
			}
		case !synthesize || hasDeveloper:
			return body, nil
		}
	}

	reqBody := map[string]any{"previous_response_id": continuation.String()}
	if instructions.Exists() {
		reqBody["instructions"] = instructions.Value()
	}
	if input.Exists() {
		var inputValue any
		if err := decodeOpenAIJSONUseNumber([]byte(input.Raw), &inputValue); err != nil {
			return body, fmt.Errorf("decode Codex input for instructions: %w", err)
		}
		reqBody["input"] = inputValue
	}
	if !normalizeCodexCacheableInstructions(reqBody, model, synthesize, compact) {
		return body, nil
	}
	if inputValue, exists := reqBody["input"]; exists {
		next, err := sjson.SetBytes(body, "input", inputValue)
		if err != nil {
			return body, err
		}
		body = next
	}
	if value, exists := reqBody["instructions"]; exists {
		return sjson.SetBytes(body, "instructions", value)
	}
	return sjson.DeleteBytes(body, "instructions")
}
