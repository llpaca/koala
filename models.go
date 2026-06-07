package main

// Models — one per provider, chosen for long context + coding strength + free tier.
//
// Free tier limits (enforced by ratelimit.go):
//   Gemini 2.5 Flash          : 10 RPM, 250 RPD, 1M context
//   Codestral                 : 2  RPM, ~unlimited RPD on experiment plan, 256k context
//   Llama 3.3 70B             : 30 RPM, 14400 RPD, 131k context
//   DeepSeek R1 Distill Llama : 30 RPM, 14400 RPD, 131k context (reasoning)

type Provider string

const (
	ProviderGoogle  Provider = "Google"
	ProviderMistral Provider = "Mistral"
	ProviderGroq    Provider = "Groq"
)

type Model struct {
	Key      string   // short identifier used in code
	ID       string   // exact model string sent to the API
	Name     string   // display name
	Provider Provider
	Endpoint string
	EnvKey   string // name of env var that holds the API key
	APIKey   string // populated at runtime from env

	// Context window (tokens) — informational
	ContextWindow int

	// Rate limits for free tier
	RPM int // requests per minute  (0 = no limit)
	RPD int // requests per day     (0 = no limit)
}

var SelectedModels = []Model{
	{
		Key:           "gemini-flash",
		ID:            "gemini-2.5-flash",
		Name:          "Gemini 2.5 Flash",
		Provider:      ProviderGoogle,
		Endpoint:      "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
		EnvKey:        "GOOGLE_API_KEY",
		ContextWindow: 1_000_000,
		RPM:           10,
		RPD:           500,
	},
	{
		Key:           "codestral",
		ID:            "codestral-latest",
		Name:          "Codestral",
		Provider:      ProviderMistral,
		Endpoint:      "https://api.mistral.ai/v1/chat/completions",
		EnvKey:        "MISTRAL_API_KEY",
		ContextWindow: 256_000,
		RPM:           2,
		RPD:           0,
	},
	{
		Key:           "llama-70b",
		ID:            "llama-3.3-70b-versatile",
		Name:          "Llama 3.3 70B",
		Provider:      ProviderGroq,
		Endpoint:      "https://api.groq.com/openai/v1/chat/completions",
		EnvKey:        "GROQ_API_KEY",
		ContextWindow: 131_072,
		RPM:           30,
		RPD:           14400,
	},
	{
		Key:           "qwen3-32b",
		ID:            "qwen/qwen3-32b",
		Name:          "Qwen3 32B",
		Provider:      ProviderGroq,
		Endpoint:      "https://api.groq.com/openai/v1/chat/completions",
		EnvKey:        "GROQ_API_KEY",
		ContextWindow: 131_072,
		RPM:           30,
		RPD:           14_400,
	},
}