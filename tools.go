package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// ── Tool definitions ──────────────────────────────────────────────────────────

// ToolDef describes a tool in a provider-agnostic way.
type ToolDef struct {
	Name        string
	Description string
	Parameters  map[string]ToolParam // name → param
	Required    []string
}

type ToolParam struct {
	Type        string // "string", "number", "boolean"
	Description string
	Enum        []string // optional allowed values
}

// AvailableTools is the master list of tools koala exposes to LLMs.
var AvailableTools = []ToolDef{
	{
		Name:        "shell",
		Description: "Execute a shell command and return stdout+stderr. Use for calculations, file listing, grepping, git commands, running scripts, etc. Commands run in the current working directory.",
		Parameters: map[string]ToolParam{
			"command": {Type: "string", Description: "The shell command to run (passed to /bin/sh -c)."},
		},
		Required: []string{"command"},
	},
	{
		Name:        "read_file",
		Description: "Read the contents of a file on disk. Returns the file text (or an error if not found). Paths can be relative or absolute.",
		Parameters: map[string]ToolParam{
			"path": {Type: "string", Description: "File path to read."},
		},
		Required: []string{"path"},
	},
	{
		Name:        "write_file",
		Description: "Write (or overwrite) a file on disk with the given text content. Creates parent directories if needed.",		Parameters: map[string]ToolParam{
			"path":    {Type: "string", Description: "File path to write."},
			"content": {Type: "string", Description: "Text content to write into the file."},
		},
		Required: []string{"path", "content"},
	},
	{
		Name:        "fetch_url",
		Description: "Fetch a URL and return the response body (up to 8 KB). Useful for reading documentation, APIs, or web pages.",
		Parameters: map[string]ToolParam{
			"url":    {Type: "string", Description: "The URL to fetch."},
			"method": {Type: "string", Description: "HTTP method: GET or POST.", Enum: []string{"GET", "POST"}},
			"body":   {Type: "string", Description: "Optional request body for POST requests."},
		},
		Required: []string{"url"},
	},
}

// ── Tool execution ────────────────────────────────────────────────────────────

// ToolCall is a parsed request from the model to invoke a tool.
type ToolCall struct {
	ID   string // provider-assigned call ID (may be empty for Gemini)
	Name string
	Args map[string]interface{}
}

// ToolResult holds the output of a tool invocation.
type ToolResult struct {
	Call   ToolCall
	Output string // either the result text or an error message
	IsErr  bool
}

// ExecuteTool runs a ToolCall and returns a ToolResult.
func ExecuteTool(call ToolCall) ToolResult {
	var out string
	var isErr bool

	defer func() {
		if r := recover(); r != nil {
			out = fmt.Sprintf("panic executing tool %s: %v", call.Name, r)
			isErr = true
		}
	}()

	switch call.Name {
	case "shell":
		out, isErr = toolShell(call.Args)
	case "read_file":
		out, isErr = toolReadFile(call.Args)
	case "write_file":
		out, isErr = toolWriteFile(call.Args)
	case "fetch_url":
		out, isErr = toolFetchURL(call.Args)
	default:
		out = fmt.Sprintf("unknown tool: %s", call.Name)
		isErr = true
	}

	return ToolResult{Call: call, Output: out, IsErr: isErr}
}

func strArg(args map[string]interface{}, key string) string {
	if v, ok := args[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

const shellTimeout = 30 * time.Second // Max duration for shell commands

func toolShell(args map[string]interface{}) (string, bool) {
	command := strArg(args, "command")
	if command == "" {
		return "error: 'command' argument is required", true
	}

	ctx_cmd := exec.Command("/bin/sh", "-c", command)
	ctx_cmd.Env = os.Environ()

	var outBuf strings.Builder
	ctx_cmd.Stdout = &outBuf
	ctx_cmd.Stderr = &outBuf

	// Hard timeout: shellTimeout
	done := make(chan error, 1)
	go func() { done <- ctx_cmd.Run() }()

	select {
	case err := <-done:
		result := outBuf.String()
		if len(result) > 6000 {
			result = result[:6000] + "\n…(truncated)"
		}
		if err != nil {
			return fmt.Sprintf("%s\n[exit: %v]", result, err), false // not a tool error — model should see exit code
		}
		return result, false
	case <-time.After(shellTimeout):
		ctx_cmd.Process.Kill()
		return fmt.Sprintf("error: command timed out after %s", shellTimeout), true
	}
}

func toolReadFile(args map[string]interface{}) (string, bool) {
	path := strArg(args, "path")
	if path == "" {
		return "error: 'path' argument is required", true
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Sprintf("error reading %q: %v", path, err), true
	}
	content := string(data)
	if len(content) > 8000 {
		content = content[:8000] + "\n…(truncated)"
	}
	return content, false
}

func toolWriteFile(args map[string]interface{}) (string, bool) {
	path := strArg(args, "path")
	content := strArg(args, "content")
	if path == "" {
		return "error: 'path' argument is required", true
	}

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Sprintf("error creating directories for %q: %v", path, err), true
	}
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return fmt.Sprintf("error writing %q: %v", path, err), true
	}
	return fmt.Sprintf("wrote %d bytes to %s", len(content), path), false
}

const fetchURLTimeout = 15 * time.Second // Max duration for fetch_url HTTP requests

var fetchClient = &http.Client{Timeout: fetchURLTimeout}

func toolFetchURL(args map[string]interface{}) (string, bool) {
	url := strArg(args, "url")
	if url == "" {
		return "error: 'url' argument is required", true
	}
	method := strArg(args, "method")
	if method == "" {
		method = "GET"
	}
	body := strArg(args, "body")

	var bodyReader io.Reader
	if body != "" {
		bodyReader = strings.NewReader(body)
	}

	req, err := http.NewRequest(method, url, bodyReader)
	if err != nil {
		return fmt.Sprintf("error creating request: %v", err), true
	}
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := fetchClient.Do(req)
	if err != nil {
		return fmt.Sprintf("error fetching %s: %v", url, err), true
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
	result := fmt.Sprintf("HTTP %d\n%s", resp.StatusCode, string(raw))
	return result, false
}

// ── Schema conversion helpers ─────────────────────────────────────────────────

// buildToolParameterSchema constructs the schema map for a single ToolParam.
// If toUpperCase is true, it converts the 'type' field to uppercase (for Gemini).
func buildToolParameterSchema(p ToolParam, toUpperCase bool) map[string]interface{} {
	paramType := p.Type
	if toUpperCase {
		paramType = strings.ToUpper(paramType)
	}
	prop := map[string]interface{}{
		"type":        paramType,
		"description": p.Description,
	}
	if len(p.Enum) > 0 {
		prop["enum"] = p.Enum
	}
	return prop
}

// toOpenAITools converts AvailableTools to OpenAI function-calling format.
func toOpenAITools() []map[string]interface{} {
	tools := make([]map[string]interface{}, 0, len(AvailableTools))
	for _, t := range AvailableTools {
		properties := map[string]interface{}{
			
		}
		for name, p := range t.Parameters {
			properties[name] = buildToolParameterSchema(p, false)
		}
		tools = append(tools, map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        t.Name,
				"description": t.Description,
				"parameters": map[string]interface{}{
					"type":       "object",
					"properties": properties,
					"required":   t.Required,
				},
			},
		})
	}
	return tools
}

// toGeminiTools converts AvailableTools to Gemini function declaration format.
func toGeminiTools() []map[string]interface{} {
	declarations := make([]map[string]interface{}, 0, len(AvailableTools))
	for _, t := range AvailableTools {
		properties := map[string]interface{}{
			
		}
		for name, p := range t.Parameters {
			properties[name] = buildToolParameterSchema(p, true)
		}
		declarations = append(declarations, map[string]interface{}{
			"name":        t.Name,
			"description": t.Description,
			"parameters": map[string]interface{}{
				"type":       "OBJECT",
				"properties": properties,
				"required":   t.Required,
			},
		})
	}
	return []map[string]interface{}{
		{"functionDeclarations": declarations},
	}
}

// parseArgs safely unmarshals a JSON-encoded args string into a map.
// It returns an error if the JSON is malformed.
func parseArgs(raw string) (map[string]interface{}, error) {
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		return nil, fmt.Errorf("failed to parse arguments JSON: %w", err)
	}
	return m, nil
}
