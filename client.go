package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

var httpClient = &http.Client{Timeout: 90 * time.Second}

// Send dispatches a request to the correct backend and returns the reply text.
func Send(m Model, history []Message) (string, error) {
	switch m.Provider {
	case ProviderMistral, ProviderGroq:
		return sendOpenAI(m, history)
	case ProviderGoogle:
		return sendGemini(m, history)
	default:
		return "", fmt.Errorf("unknown provider: %s", m.Provider)
	}
}

// ── OpenAI-compatible (Mistral, Groq) ───────────────────────────────────────

type openAIRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

type openAIResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func sendOpenAI(m Model, history []Message) (string, error) {
	body, _ := json.Marshal(openAIRequest{
		Model:    m.ID,
		Messages: history,
		Stream:   false,
	})

	req, err := http.NewRequest("POST", m.Endpoint, bytes.NewBuffer(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.APIKey)

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, trimBody(raw))
	}

	var result openAIResponse
	if err := json.Unmarshal(raw, &result); err != nil {
		return "", fmt.Errorf("decode error: %w", err)
	}
	if result.Error != nil {
		return "", fmt.Errorf("API error: %s", result.Error.Message)
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("empty response from %s", m.Name)
	}
	return result.Choices[0].Message.Content, nil
}

// ── Google Gemini ────────────────────────────────────────────────────────────

type geminiPart struct {
	Text string `json:"text"`
}
type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}
type geminiRequest struct {
	Contents []geminiContent `json:"contents"`
}
type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []geminiPart `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func sendGemini(m Model, history []Message) (string, error) {
	var contents []geminiContent
	var systemBuf string

	for _, msg := range history {
		switch msg.Role {
		case "system":
			systemBuf = msg.Content
			continue
		case "assistant":
			msg.Role = "model"
		}

		text := msg.Content
		if msg.Role == "user" && systemBuf != "" {
			text = systemBuf + "\n\n" + text
			systemBuf = ""
		}
		contents = append(contents, geminiContent{
			Role:  msg.Role,
			Parts: []geminiPart{{Text: text}},
		})
	}

	if len(contents) == 0 {
		return "", fmt.Errorf("no messages to send")
	}

	body, _ := json.Marshal(geminiRequest{Contents: contents})

	url := fmt.Sprintf("%s?key=%s", m.Endpoint, m.APIKey)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, trimBody(raw))
	}

	var result geminiResponse
	if err := json.Unmarshal(raw, &result); err != nil {
		return "", fmt.Errorf("decode error: %w", err)
	}
	if result.Error != nil {
		return "", fmt.Errorf("API error: %s", result.Error.Message)
	}
	if len(result.Candidates) == 0 || len(result.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("empty response from %s", m.Name)
	}
	return result.Candidates[0].Content.Parts[0].Text, nil
}

func trimBody(b []byte) string {
	if len(b) > 400 {
		return string(b[:400]) + "…"
	}
	return string(b)
}
