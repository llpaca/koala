package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

var httpClient = &http.Client{Timeout: 90 * time.Second}

const maxToolIterations = 10 // Max attempts for the model to self-correct with tools

// Send dispatches a request to the correct backend and runs the tool-call loop.
// It uses pool to select an API key, retrying with the next key in the pool
// if the current one reports a quota/rate-limit error.
func Send(m Model, history []Message, pool *KeyPool) (string, error) {
	var lastErr error
	for attempt := 0; attempt < len(pool.Entries); attempt++ {
		entry, err := pool.Next()
		if err != nil {
			if lastErr != nil {
				return "", lastErr
			}
			return "", err
		}

		mm := m
		mm.APIKey = entry.APIKey

		if err = entry.Limiter.Wait(); err != nil {
			entry.Limiter.MarkExhausted()
			lastErr = fmt.Errorf("rate limit: %w", err)
			printWarn(fmt.Sprintf("%s key %s: %v, trying next key…", mm.Name, entry.EnvVar, err))
			continue
		}

		var reply string
		switch mm.Provider {
		case ProviderMistral, ProviderGroq:
			reply, err = sendOpenAI(mm, history)
		case ProviderGoogle:
			reply, err = sendGemini(mm, history)
		default:
			return "", fmt.Errorf("unknown provider: %s", mm.Provider)
		}

		if err == nil {
			return reply, nil
		}

		if isQuotaError(err) {
			entry.Limiter.MarkExhausted()
			lastErr = err
			printWarn(fmt.Sprintf("%s key %s exhausted, trying next key…", mm.Name, entry.EnvVar))
			continue
		}

		return "", err
	}

	if lastErr != nil {
		return "", lastErr
	}
	return "", fmt.Errorf("no usable API key for %s", m.Name)
}

// isQuotaError detects provider responses indicating quota/rate-limit exhaustion.
func isQuotaError(err error) bool {
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "quota") ||
		strings.Contains(msg, "resource_exhausted") ||
		strings.Contains(msg, "rate limit") ||
		strings.Contains(msg, "429") ||
		strings.Contains(msg, "403") ||
		strings.Contains(msg, "permission_denied") ||
		strings.Contains(msg, "denied access")
}

// doRequestAndUnmarshal sends an HTTP request and unmarshals the JSON response.
func doRequestAndUnmarshal(method, url string, headers map[string]string, reqBody interface{}, respTarget interface{}) error {
	var bodyReader io.Reader
	if reqBody != nil {
		jsonBody, err := json.Marshal(reqBody)
		if err != nil {
			return fmt.Errorf("marshal request body: %w", err)
		}
		bodyReader = bytes.NewBuffer(jsonBody)
	}

	req, err := http.NewRequest(method, url, bodyReader)
	if err != nil {
		return err
	}

	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, trimBody(raw))
	}

	if respTarget != nil {
		if err := json.Unmarshal(raw, respTarget); err != nil {
			return fmt.Errorf("decode response error: %w", err)
		}
	}
	return nil
}

// executeToolCalls runs a slice of ToolCall's and returns their results.
func executeToolCalls(m Model, calls []ToolCall) []ToolResult {
	var results []ToolResult
	for _, call := range calls {
		printToolCall(m.Provider, call)
		res := ExecuteTool(call)
		printToolResult(m.Provider, res)
		results = append(results, res)
	}
	return results
}

// ── OpenAI-compatible (Mistral, Groq) ────────────────────────────────────────

type openAIMessage struct {
	Role       string            `json:"role"`
	Content    interface{}       `json:"content"` // string or null
	ToolCalls  []openAIToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string            `json:"tool_call_id,omitempty"`
	Name       string            `json:"name,omitempty"`
}

type openAIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type openAIRequest struct {
	Model    string          `json:"model"`
	Messages []openAIMessage `json:"messages"`
	Tools    interface{}     `json:"tools,omitempty"`
	Stream   bool            `json:"stream"`
}

type openAIResponse struct {
	Choices []struct {
		Message      openAIMessage `json:"message"`
		FinishReason string        `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func messagesForOpenAI(history []Message) []openAIMessage {
	out := make([]openAIMessage, 0, len(history))
	for _, m := range history {
		out = append(out, openAIMessage{Role: m.Role, Content: m.Content})
	}
	return out
}

func sendOpenAI(m Model, history []Message) (string, error) {
	msgs := messagesForOpenAI(history)
	tools := toOpenAITools()

	for iter := 0; iter < maxToolIterations; iter++ {
		reqBody := openAIRequest{
			Model:    m.ID,
			Messages: msgs,
			Tools:    tools,
			Stream:   false,
		}
		headers := map[string]string{
			"Content-Type":  "application/json",
			"Authorization": "Bearer " + m.APIKey,
		}

		var result openAIResponse
		if err := doRequestAndUnmarshal("POST", m.Endpoint, headers, reqBody, &result); err != nil {
			return "", err
		}

		if result.Error != nil {
			return "", fmt.Errorf("API error: %s", result.Error.Message)
		}
		if len(result.Choices) == 0 {
			return "", fmt.Errorf("empty response from %s", m.Name)
		}

		choice := result.Choices[0]
		assistantMsg := choice.Message

		// No tool calls → final answer
		if len(assistantMsg.ToolCalls) == 0 {
			content := ""
			if s, ok := assistantMsg.Content.(string); ok {
				content = s
			}
			return content, nil
		}

		// Add assistant message (with tool_calls) to conversation
		msgs = append(msgs, assistantMsg)

		// Convert to generic ToolCall and execute
		var calls []ToolCall
		for _, tc := range assistantMsg.ToolCalls {
			args, err := parseArgs(tc.Function.Arguments)
			if err != nil {
				return "", fmt.Errorf("tool call argument parsing failed for %s: %w", tc.Function.Name, err)
			}
			calls = append(calls, ToolCall{
				ID:   tc.ID,
				Name: tc.Function.Name,
				Args: args,
			})
		}
		toolResults := executeToolCalls(m, calls)

		// Append tool results to conversation
		for _, res := range toolResults {
			msgs = append(msgs, openAIMessage{
				Role:       "tool",
				ToolCallID: res.Call.ID,
				Name:       res.Call.Name,
				Content:    res.Output,
			})
		}
	}

	return "", fmt.Errorf("exceeded max tool iterations (%d)", maxToolIterations)
}

// ── Google Gemini ─────────────────────────────────────────────────────────────

type geminiPart struct {
	Text             string                 `json:"text,omitempty"`
	FunctionCall     *geminiFunctionCall    `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResp    `json:"functionResponse,omitempty"`
}
type geminiFunctionCall struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args"`
}
type geminiFunctionResp struct {
	Name     string                 `json:"name"`
	Response map[string]interface{} `json:"response"`
}
type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}
type geminiRequest struct {
	Contents []geminiContent        `json:"contents"`
	Tools    []map[string]interface{} `json:"tools,omitempty"`
}
type geminiResponse struct {
	Candidates []struct {
		Content      geminiContent `json:"content"`
		FinishReason string        `json:"finishReason"`
	} `json:"candidates"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func historyToGemini(history []Message) ([]geminiContent, error) {
	var contents []geminiContent
	var systemInstructions []string
	var filteredHistory []Message

	// First pass: collect all system messages and filter them out
	for _, msg := range history {
		if msg.Role == "system" {
			systemInstructions = append(systemInstructions, msg.Content)
		} else {
			filteredHistory = append(filteredHistory, msg)
		}
	}

	combinedSystemInstruction := strings.Join(systemInstructions, "\n\n")

	// Second pass: build geminiContent, prepending system instruction to the first user message
	systemInstructionAdded := false
	for _, msg := range filteredHistory {
		role := msg.Role
		if role == "assistant" {
			role = "model" // Gemini expects "model" for assistant roles
		}

		text := msg.Content
		if role == "user" && !systemInstructionAdded && combinedSystemInstruction != "" {
			text = combinedSystemInstruction + "\n\n" + text
			systemInstructionAdded = true
		}

		contents = append(contents, geminiContent{
			Role:  role,
			Parts: []geminiPart{{Text: text}},
		})
	}

	if len(contents) == 0 {
		// If there were only system messages, or an empty history
		if combinedSystemInstruction != "" {
			// If only system instructions, and no user messages to prepend to,
			// we create an initial user message for it.
			contents = append(contents, geminiContent{
				Role:  "user",
				Parts: []geminiPart{{Text: combinedSystemInstruction}},
			})
		} else {
			return nil, fmt.Errorf("no messages to send")
		}
	}

	return contents, nil
}

func sendGemini(m Model, history []Message) (string, error) {
	contents, err := historyToGemini(history)
	if err != nil {
		return "", err
	}
	tools := toGeminiTools()

	for iter := 0; iter < maxToolIterations; iter++ {
		reqBody := geminiRequest{Contents: contents, Tools: tools}
		url := fmt.Sprintf("%s?key=%s", m.Endpoint, m.APIKey)
		headers := map[string]string{
			"Content-Type": "application/json",
		}

		var result geminiResponse
		if err := doRequestAndUnmarshal("POST", url, headers, reqBody, &result); err != nil {
			return "", err
		}

		if result.Error != nil {
			return "", fmt.Errorf("API error: %s", result.Error.Message)
		}
		if len(result.Candidates) == 0 {
			return "", fmt.Errorf("empty response from %s", m.Name)
		}

		candidate := result.Candidates[0]
		candidateContent := candidate.Content

		// Collect function calls from this turn
		var funcCalls []geminiPart
		var textParts []string
		for _, part := range candidateContent.Parts {
			if part.FunctionCall != nil {
				funcCalls = append(funcCalls, part)
			} else if part.Text != "" {
				textParts = append(textParts, part.Text)
			}
		}

		// No function calls → final answer
		if len(funcCalls) == 0 {
			if len(textParts) == 0 {
				return "", fmt.Errorf("empty response from %s", m.Name)
			}
			result := ""
			for _, t := range textParts {
				result += t
			}
			return result, nil
		}

		// Append model's turn (with function calls)
		contents = append(contents, candidateContent)

		// Convert to generic ToolCall and execute
		var calls []ToolCall
		for _, part := range funcCalls {
			fc := part.FunctionCall
			calls = append(calls, ToolCall{Name: fc.Name, Args: fc.Args})
		}
		toolResults := executeToolCalls(m, calls)

		// Build the "user" function-response turn
		var responseParts []geminiPart
		for _, res := range toolResults {
			responseParts = append(responseParts, geminiPart{
				FunctionResponse: &geminiFunctionResp{
					Name:     res.Call.Name,
					Response: map[string]interface{}{"output": res.Output},
				},
			})
		}
		contents = append(contents, geminiContent{
			Role:  "user",
			Parts: responseParts,
		})
	}

	return "", fmt.Errorf("exceeded max tool iterations (%d)", maxToolIterations)
}

func trimBody(b []byte) string {
	if len(b) > 400 {
		return string(b[:400]) + "…"
	}
	return string(b)
}
