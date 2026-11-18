package service

import (
	"bytes"
	"fmt"
	"strings"
	"text/template"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai"
)

type forcedCodexInstructionsTemplateData struct {
	ExistingInstructions string
	OriginalModel        string
	NormalizedModel      string
	BillingModel         string
	UpstreamModel        string
}

func applyForcedCodexInstructionsTemplate(
	reqBody map[string]any,
	templateText string,
	data forcedCodexInstructionsTemplateData,
) (bool, error) {
	input, _ := reqBody["input"].([]any)
	continuation, _ := reqBody["previous_response_id"].(string)
	usePrefix := openai.GPT62ModelID(data.UpstreamModel) != "" && strings.TrimSpace(continuation) == ""
	var replaceLeading bool
	if usePrefix {
		data.ExistingInstructions, replaceLeading = leadingCodexDeveloperInstructions(input)
	}
	rendered, err := renderForcedCodexInstructionsTemplate(templateText, data)
	if err != nil {
		return false, err
	}
	if rendered == "" {
		return false, nil
	}
	if usePrefix {
		if replaceLeading {
			if data.ExistingInstructions == rendered {
				return false, nil
			}
			message := input[0].(map[string]any)
			message["content"] = []any{map[string]any{"type": "input_text", "text": rendered}}
		} else {
			input = append([]any{codexDeveloperMessage(rendered)}, input...)
		}
		reqBody["input"] = input
		return true, nil
	}

	existing, _ := reqBody["instructions"].(string)
	if strings.TrimSpace(existing) == rendered {
		return false, nil
	}

	reqBody["instructions"] = rendered
	return true, nil
}

func renderForcedCodexInstructionsTemplate(
	templateText string,
	data forcedCodexInstructionsTemplateData,
) (string, error) {
	tmpl, err := template.New("forced_codex_instructions").Option("missingkey=zero").Parse(templateText)
	if err != nil {
		return "", fmt.Errorf("parse forced codex instructions template: %w", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("render forced codex instructions template: %w", err)
	}

	return strings.TrimSpace(buf.String()), nil
}
