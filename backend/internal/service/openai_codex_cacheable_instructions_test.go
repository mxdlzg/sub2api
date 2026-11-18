package service

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestGPT62CodexResponsesCacheableBaseInstructions(t *testing.T) {
	gin.SetMode(gin.TestMode)
	const model = "gpt-6.2-astra"
	const completed = `data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":2,"output_tokens":1}}}`
	base := strings.ReplaceAll(defaultCodexSynthInstructions(model), "{{connector_id}}", "example_connector")
	for _, tc := range []struct {
		name        string
		passthrough bool
	}{
		{name: "transform"},
		{name: "passthrough", passthrough: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var prefix string
			for _, instructions := range []string{"turn one", "turn two"} {
				body, err := json.Marshal(map[string]any{
					"model":        model,
					"input":        "Hello",
					"instructions": base + "\n\n" + instructions,
					"stream":       true,
				})
				require.NoError(t, err)
				recorder := httptest.NewRecorder()
				c, _ := gin.CreateTestContext(recorder)
				c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
				c.Request.Header.Set("User-Agent", "codex_cli_rs/0.98.0")
				upstream := &httpUpstreamRecorder{resp: &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": {"text/event-stream"}},
					Body:       io.NopCloser(strings.NewReader(completed + "\n\n")),
				}}
				svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
				account := &Account{
					ID:          123,
					Platform:    PlatformOpenAI,
					Type:        AccountTypeOAuth,
					Concurrency: 1,
					Credentials: map[string]any{"access_token": "oauth-token", "chatgpt_account_id": "chatgpt-acc"},
					Extra: map[string]any{
						"openai_passthrough":                        tc.passthrough,
						"openai_oauth_responses_websockets_v2_mode": OpenAIWSIngressModeOff,
					},
				}
				_, err = svc.Forward(context.Background(), c, account, body)
				require.NoError(t, err)
				out := gjson.ParseBytes(upstream.lastBody)
				require.Equal(t, instructions, out.Get("instructions").String())
				require.Equal(t, "developer", out.Get("input.0.role").String())
				require.Equal(t, base, out.Get("input.0.content.0.text").String())
				require.Equal(t, "user", out.Get("input.1.role").String())
				if prefix != "" {
					require.JSONEq(t, prefix, out.Get("input.0").Raw)
				}
				prefix = out.Get("input.0").Raw
			}
		})
	}
}

func TestGPT62CodexInstructionPlacement(t *testing.T) {
	const model = "gpt-6.2-sol"
	base := defaultCodexSynthInstructions(model)

	t.Run("default prefix is idempotent", func(t *testing.T) {
		req := map[string]any{"model": model, "input": "hello"}
		require.NoError(t, applyCodexOAuthTransform(req, false, false).Error)
		require.NoError(t, applyCodexOAuthTransform(req, false, false).Error)
		require.Equal(t, "", req["instructions"])
		input, ok := req["input"].([]any)
		require.True(t, ok)
		require.Len(t, input, 2)
		require.Equal(t, codexDeveloperMessage(base), input[0])
	})

	t.Run("native developer prefix does not trigger a fallback prompt", func(t *testing.T) {
		req := map[string]any{
			"model": model,
			"input": []any{codexDeveloperMessage("Codex client base")},
		}
		require.NoError(t, applyCodexOAuthTransform(req, true, false).Error)
		require.Equal(t, []any{codexDeveloperMessage("Codex client base")}, req["input"])
		require.NotContains(t, req, "instructions")
	})

	t.Run("input system stays in order and turn instructions stay separate", func(t *testing.T) {
		req := map[string]any{
			"model": model, "instructions": "This turn only",
			"input": []any{
				map[string]any{"type": "message", "role": "system", "content": "Stable system"},
				map[string]any{"type": "message", "role": "user", "content": "hello"},
				codexDeveloperMessage("Later developer update"),
			},
		}
		result := applyCodexOAuthTransformWithOptions(req, codexOAuthTransformOptions{OmitPromotedSystemMessagesFromInput: true})
		require.NoError(t, result.Error)
		require.Equal(t, "This turn only", req["instructions"])
		input, ok := req["input"].([]any)
		require.True(t, ok)
		require.Len(t, input, 3)
		first, ok := input[0].(map[string]any)
		require.True(t, ok)
		require.Equal(t, "developer", first["role"])
		require.Equal(t, codexDeveloperMessage("Later developer update"), input[2])
	})

	t.Run("existing base is not repeated", func(t *testing.T) {
		req := map[string]any{"model": model, "input": []any{codexDeveloperMessage(base)}, "instructions": base}
		require.NoError(t, applyCodexOAuthTransform(req, true, false).Error)
		require.Len(t, req["input"], 1)
		require.Equal(t, "", req["instructions"])
	})

	t.Run("incremental requests preserve instructions", func(t *testing.T) {
		req := map[string]any{"model": model, "previous_response_id": "resp_1", "instructions": base, "input": "hello"}
		require.NoError(t, applyCodexOAuthTransform(req, true, false).Error)
		require.Len(t, req["input"], 1)
		require.Equal(t, base, req["instructions"])
	})

	t.Run("compact omits instructions", func(t *testing.T) {
		req := map[string]any{"model": model, "input": "hello", "instructions": base}
		require.NoError(t, applyCodexOAuthTransform(req, true, true).Error)
		require.Len(t, req["input"], 1)
		require.NotContains(t, req, "instructions")
	})

	t.Run("compacted replay does not gain a fallback prompt", func(t *testing.T) {
		body := []byte(`{"model":"gpt-6.2-sol","input":[{"type":"compaction","encrypted_content":"opaque"},{"type":"message","role":"user","content":"hello"}]}`)
		out, err := normalizeCodexCacheableInstructionsBody(body, model, true, false)
		require.NoError(t, err)
		require.Equal(t, body, out)
	})

	t.Run("template uses the existing developer prefix", func(t *testing.T) {
		req := map[string]any{
			"input":        []any{codexDeveloperMessage("Client prefix"), codexDeveloperMessage("Later developer update")},
			"instructions": "This turn only",
		}
		changed, err := applyForcedCodexInstructionsTemplate(req, "Server prefix\n\n{{ .ExistingInstructions }}", forcedCodexInstructionsTemplateData{UpstreamModel: model})
		require.NoError(t, err)
		require.True(t, changed)
		require.Equal(t, "This turn only", req["instructions"])
		require.Equal(t, []any{codexDeveloperMessage("Server prefix\n\nClient prefix"), codexDeveloperMessage("Later developer update")}, req["input"])
	})
}
