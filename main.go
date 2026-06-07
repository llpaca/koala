package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

func main() {
	// Load .env
	if err := LoadEnv(".env"); err != nil && !os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "warning: .env load error: %v\n", err)
	}

	// Inject API keys into model definitions
	for i := range SelectedModels {
		SelectedModels[i].APIKey = os.Getenv(SelectedModels[i].EnvKey)
	}

	// Build rate limiters for every model
	limiters := make(map[string]*RateLimiter)
	for _, m := range SelectedModels {
		limiters[m.Key] = NewRateLimiter(m.RPM, m.RPD)
	}

	// Find which models are actually usable (have an API key)
	var available []Model
	for _, m := range SelectedModels {
		if m.APIKey != "" {
			available = append(available, m)
		}
	}

	printHeader()

	if len(available) == 0 {
		fmt.Println(Red + "  No API keys found. Create a .env file with:" + Reset)
		fmt.Println(Grey + "    GOOGLE_API_KEY=...")
		fmt.Println("    MISTRAL_API_KEY=...")
		fmt.Println("    GROQ_API_KEY=..." + Reset)
		os.Exit(1)
	}

	printModelList(available, limiters)

	reader := bufio.NewReader(os.Stdin)

	// ── Mode selection ────────────────────────────────────────────────────────
	fmt.Println(Bold + "  Chat mode:" + Reset)
	for i, m := range available {
		col := providerColor(m.Provider)
		fmt.Printf("  %s%d.%s %sSingle chat → %s%s\n", Bold, i+1, Reset, col, m.Name, Reset)
	}
	fmt.Printf("  %s%d.%s %sGroup chat (broadcast to all)%s\n",
		Bold, len(available)+1, Reset, Magenta, Reset)
	fmt.Println()
	fmt.Print(Bold + "  Select mode: " + Reset)

	modeInput := readLine(reader)
	modeIdx := 0
	fmt.Sscanf(modeInput, "%d", &modeIdx)

	fmt.Println()

	if modeIdx >= 1 && modeIdx <= len(available) {
		// Single model chat
		selected := available[modeIdx-1]
		runSingleChat(reader, selected, limiters[selected.Key])
	} else if modeIdx == len(available)+1 {
		// Group chat
		runGroupChat(reader, available, limiters)
	} else {
		fmt.Println(Red + "  Invalid selection." + Reset)
		os.Exit(1)
	}
}

// ── Single chat ───────────────────────────────────────────────────────────────

func runSingleChat(reader *bufio.Reader, m Model, limiter *RateLimiter) {
	conv := NewConversation()
	spinner := NewSpinner()

	col := providerColor(m.Provider)
	fmt.Printf("%s%s  Chatting with %s%s\n", col, Bold, m.Name, Reset)
	printHelp()

	for {
		fmt.Printf("%s%s›%s ", col, Bold, Reset)
		prompt := readLine(reader)

		if handleMeta(prompt, conv) {
			continue
		}
		if prompt == "quit" || prompt == "exit" || prompt == "q" {
			break
		}
		if prompt == "" {
			continue
		}

		conv.Add("user", prompt)
		spinner.Start(fmt.Sprintf("waiting for %s…", m.Name))

		start := time.Now()
		reply, err := sendWithLimit(m, limiter, conv.Get())
		elapsed := time.Since(start)

		spinner.Stop()

		if err != nil {
			printError(m, err)
			// Remove the user message we just added so history stays clean
			conv.Clear()
			for _, msg := range conv.Get() {
				conv.Add(msg.Role, msg.Content)
			}
			continue
		}

		conv.Add("assistant", reply)
		printResponse(m, reply, elapsed)
	}
}

// ── Group chat ────────────────────────────────────────────────────────────────

func runGroupChat(reader *bufio.Reader, models []Model, limiters map[string]*RateLimiter) {
	// Each model gets its own conversation history
	convs := make(map[string]*Conversation)
	for _, m := range models {
		convs[m.Key] = NewConversation()
	}

	spinner := NewSpinner()

	fmt.Printf("%s%s  Group chat — broadcasting to %d models%s\n",
		Magenta, Bold, len(models), Reset)
	printHelp()

	for {
		fmt.Printf("%s%s›%s ", Magenta, Bold, Reset)
		prompt := readLine(reader)

		if prompt == "quit" || prompt == "exit" || prompt == "q" {
			break
		}
		// Meta commands apply to all conversations
		if prompt == "reset" {
			for _, c := range convs {
				c.Clear()
			}
			printInfo("All conversation histories cleared.")
			continue
		}
		if prompt == "help" {
			printHelp()
			continue
		}
		if prompt == "" {
			continue
		}

		// Add user message to every conversation
		for _, c := range convs {
			c.Add("user", prompt)
		}

		// Fan out requests concurrently
		type result struct {
			model   Model
			reply   string
			elapsed time.Duration
			err     error
		}

		results := make(chan result, len(models))
		var wg sync.WaitGroup

		spinner.Start(fmt.Sprintf("waiting for %d models…", len(models)))

		for _, m := range models {
			wg.Add(1)
			go func(m Model) {
				defer wg.Done()
				start := time.Now()
				reply, err := sendWithLimit(m, limiters[m.Key], convs[m.Key].Get())
				results <- result{m, reply, time.Since(start), err}
			}(m)
		}

		// Close channel when all goroutines finish
		go func() {
			wg.Wait()
			close(results)
		}()

		// Collect and display in arrival order
		var ordered []result
		for r := range results {
			ordered = append(ordered, r)
		}
		spinner.Stop()

		// Sort by elapsed for nicer display (fastest first)
		// Simple insertion sort — only 3–4 items
		for i := 1; i < len(ordered); i++ {
			for j := i; j > 0 && ordered[j].elapsed < ordered[j-1].elapsed; j-- {
				ordered[j], ordered[j-1] = ordered[j-1], ordered[j]
			}
		}

		for _, r := range ordered {
			if r.err != nil {
				printError(r.model, r.err)
				// Roll back user message for this model
				convs[r.model.Key].Clear()
			} else {
				convs[r.model.Key].Add("assistant", r.reply)
				printResponse(r.model, r.reply, r.elapsed)
			}
		}
	}
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// sendWithLimit applies rate limiting then sends the request.
func sendWithLimit(m Model, limiter *RateLimiter, history []Message) (string, error) {
	if err := limiter.Wait(); err != nil {
		return "", fmt.Errorf("rate limit: %w", err)
	}
	return Send(m, history)
}

// handleMeta processes commands that are not chat messages.
// Returns true if the prompt was handled (should not be sent to LLM).
func handleMeta(prompt string, conv *Conversation) bool {
	switch strings.ToLower(prompt) {
	case "reset", "clear history":
		conv.Clear()
		printInfo("Conversation history cleared.")
		return true
	case "help":
		printHelp()
		return true
	case "status":
		printInfo(fmt.Sprintf("History: %d messages", conv.Len()))
		return true
	}
	return false
}

func printHelp() {
	fmt.Printf("%s  Commands: reset · status · help · quit%s\n\n", Grey, Reset)
}

func readLine(r *bufio.Reader) string {
	line, _ := r.ReadString('\n')
	return strings.TrimSpace(line)
}
