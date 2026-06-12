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

	// Build a key pool for every model (handles multi-key rotation, e.g. Gemini)
	pools := make(map[string]*KeyPool)
	for _, m := range SelectedModels {
		pools[m.Key] = NewKeyPool(m)
	}

	// Find which models are actually usable (have at least one API key)
	var available []Model
	for _, m := range SelectedModels {
		if pools[m.Key].Available() {
			available = append(available, m)
		}
	}

	printHeader()

	if len(available) == 0 {
		fmt.Println(Red + "  No API keys found. Create a .env file with:" + Reset)
		fmt.Println(Grey + "    GOOGLE_API_KEY_1=...")
		fmt.Println("    MISTRAL_API_KEY=...")
		fmt.Println("    GROQ_API_KEY=..." + Reset)
		os.Exit(1)
	}

	printModelList(available, pools)

	reader := bufio.NewReader(os.Stdin)

	// ── Mode selection ────────────────────────────────────────────────────────
	fmt.Println(Bold + "  Chat mode:" + Reset)
	for i, m := range available {
		col := providerColor(m.Provider)
		fmt.Printf("  %s%d.%s %sSingle chat → %s%s\n", Bold, i+1, Reset, col, m.Name, Reset)
	}
	groupIdx := len(available) + 1
	selfIdx := len(available) + 2
	fmt.Printf("  %s%d.%s %sGroup chat (broadcast to all)%s\n",
		Bold, groupIdx, Reset, Magenta, Reset)
	fmt.Printf("  %s%d.%s %sSelf-improvement loop (agent rewrites itself)%s\n",
		Bold, selfIdx, Reset, Yellow, Reset)
	fmt.Println()
	fmt.Print(Bold + "  Select mode: " + Reset)

	modeInput, err := readLine(reader)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%sError reading input: %v%s\n", Red, err, Reset)
		os.Exit(1)
	}

	modeIdx := 0
	fmt.Sscanf(modeInput, "%d", &modeIdx)

	fmt.Println()

	if modeIdx >= 1 && modeIdx <= len(available) {
		// Single model chat
		selected := available[modeIdx-1]
		runSingleChat(reader, selected, pools[selected.Key])
	} else if modeIdx == groupIdx {
		// Group chat
		runGroupChat(reader, available, pools)
	} else if modeIdx == selfIdx {
		// Task or self-improvement loop
		fmt.Print(Bold + "  Task (leave blank for self-improvement mode):\n  › " + Reset)
		task, err := readLine(reader)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sError reading input: %v%s\n", Red, err, Reset)
			os.Exit(1)
		}
		fmt.Println()
		RunSelfLoop(available, pools, 0, task)
	} else {
		fmt.Println(Red + "  Invalid selection." + Reset)
		os.Exit(1)
	}
}

// ── Single chat ───────────────────────────────────────────────────────────────

func runSingleChat(reader *bufio.Reader, m Model, pool *KeyPool) {
	conv := NewConversation()
	spinner := NewSpinner()

	col := providerColor(m.Provider)
	fmt.Printf("%s%s  Chatting with %s%s\n", col, Bold, m.Name, Reset)
	printHelp()

	for {
		fmt.Printf("%s%s›%s ", col, Bold, Reset)
		prompt, err := readLine(reader)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sError reading input: %v%s\n", Red, err, Reset)
			break // Exit chat loop on input error
		}

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
		reply, err := sendWithLimit(m, pool, conv.Get())
		elapsed := time.Since(start)

		spinner.Stop()

		if err != nil {
			printError(m.Provider, m.Name, err)
			// Remove the user message we just added so history stays clean
			conv.Clear()
			for _, msg := range conv.Get() {
				conv.Add(msg.Role, msg.Content)
			}
			continue
		}

		conv.Add("assistant", reply)
		printResponse(m.Provider, m.Name, m.ID, reply, elapsed)
	}
}

// ── Group chat ────────────────────────────────────────────────────────────────

func runGroupChat(reader *bufio.Reader, models []Model, pools map[string]*KeyPool) {
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
		prompt, err := readLine(reader)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sError reading input: %v%s\n", Red, err, Reset)
			break // Exit chat loop on input error
		}

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
				reply, err := sendWithLimit(m, pools[m.Key], convs[m.Key].Get())
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
				printError(r.model.Provider, r.model.Name, r.err)
				// Roll back user message for this model
				convs[r.model.Key].Clear()
			} else {
				convs[r.model.Key].Add("assistant", r.reply)
				printResponse(r.model.Provider, r.model.Name, r.model.ID, r.reply, r.elapsed)
			}
		}
	}
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// sendWithLimit dispatches the request via the model's key pool.
func sendWithLimit(m Model, pool *KeyPool, history []Message) (string, error) {
	return Send(m, history, pool)
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

func readLine(r *bufio.Reader) (string, error) {
	line, err := r.ReadString('\n')
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(line), nil
}
