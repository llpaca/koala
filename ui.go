package main

import (
	"fmt"
	"strings"
	"sync"
	"time"
)

// ── ANSI ─────────────────────────────────────────────────────────────────────

const (
	Reset   = "\033[0m"
	Bold    = "\033[1m"
	Dim     = "\033[2m"
	Red     = "\033[31m"
	Green   = "\033[32m"
	Yellow  = "\033[33m"
	Blue    = "\033[34m"
	Magenta = "\033[35m"
	Cyan    = "\033[36m"
	White   = "\033[37m"
	Grey    = "\033[90m"

	BgBlue = "\033[44m"
	BgGrey = "\033[100m"

	ClearLine = "\033[2K\r"
	Up        = "\033[1A"
)

// providerColor returns a consistent color for each provider.
func providerColor(p Provider) string {
	switch p {
	case ProviderGoogle:
		return Blue
	case ProviderMistral:
		return Magenta
	case ProviderGroq:
		return Cyan
	default:
		return White
	}
}

// modelTag returns a colored "[Name]" label for a model.
func modelTag(m Model) string {
	col := providerColor(m.Provider)
	return fmt.Sprintf("%s%s[%s]%s", col, Bold, m.Name, Reset)
}

// ── Spinner ──────────────────────────────────────────────────────────────────

type Spinner struct {
	mu      sync.Mutex
	running bool
	frames  []string
	label   string
}

func NewSpinner() *Spinner {
	return &Spinner{
		frames: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"},
	}
}

func (s *Spinner) Start(label string) {
	s.mu.Lock()
	s.running = true
	s.label = label
	s.mu.Unlock()

	go func() {
		i := 0
		for {
			s.mu.Lock()
			if !s.running {
				fmt.Print(ClearLine)
				s.mu.Unlock()
				return
			}
			frame := s.frames[i%len(s.frames)]
			lbl := s.label
			s.mu.Unlock()

			fmt.Printf("%s%s%s %s%s", ClearLine, Grey, frame, lbl, Reset)
			time.Sleep(80 * time.Millisecond)
			i++
		}
	}()
}

func (s *Spinner) SetLabel(label string) {
	s.mu.Lock()
	s.label = label
	s.mu.Unlock()
}

func (s *Spinner) Stop() {
	s.mu.Lock()
	s.running = false
	s.mu.Unlock()
	time.Sleep(100 * time.Millisecond) // let goroutine clear the line
}

// ── Print helpers ─────────────────────────────────────────────────────────────

func printHeader() {
	fmt.Print("\033[H\033[2J") // clear screen
	line := strings.Repeat("─", 60)
	fmt.Printf("\n%s%s%s\n", Cyan+Bold, "  llmtui — multi-provider LLM terminal client", Reset)
	fmt.Printf("%s%s%s\n\n", Grey, line, Reset)
}

func printModelList(models []Model, limiters map[string]*RateLimiter) {
	fmt.Println(Bold + "  Models:" + Reset)
	for i, m := range models {
		col := providerColor(m.Provider)
		status := ""
		if l, ok := limiters[m.Key]; ok {
			status = Grey + " · " + l.Status() + Reset
		}
		ctxK := m.ContextWindow / 1000
		fmt.Printf("  %s%d.%s %s%-18s%s %s(%s)%s  %s%dk ctx%s%s\n",
			Bold, i+1, Reset,
			col+Bold, m.Name, Reset,
			Grey, m.Provider, Reset,
			Yellow, ctxK, Reset,
			status,
		)
	}
	fmt.Println()
}

func printDivider() {
	fmt.Printf("%s%s%s\n", Grey, strings.Repeat("─", 60), Reset)
}

func printResponse(m Model, text string, elapsed time.Duration) {
	col := providerColor(m.Provider)
	fmt.Printf("\n%s%s▶ %s%s\n", col, Bold, m.Name, Reset)
	printDivider()
	fmt.Println(text)
	fmt.Printf("%s[%.2fs · %s · %s]%s\n\n",
		Grey, elapsed.Seconds(), m.Provider, m.ID, Reset)
}

func printError(m Model, err error) {
	fmt.Printf("\n%s%s✖ %s:%s %v\n\n", Red, Bold, m.Name, Reset, err)
}

func printInfo(msg string) {
	fmt.Printf("%sℹ %s%s\n", Cyan, msg, Reset)
}

func printWarn(msg string) {
	fmt.Printf("%s⚠ %s%s\n", Yellow, msg, Reset)
}
