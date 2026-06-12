package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"
)

// ── Constants ─────────────────────────────────────────────────────────────────

const memoryFile    = "memory.md"
const maxFixAttempts = 3

// switchThreshold — switch away from a model when daily headroom drops below this.
const switchThreshold = 0.15

// Source files the agent is allowed to read and edit.
var sourceFiles = []string{
	"main.go",
	"client.go",
	"config.go",
	"conversation.go",
	"keypool.go",
	"models.go",
	"ratelimit.go",
	"tools.go",
	"ui.go",
	"self.go",
}

// ── Model orchestration ───────────────────────────────────────────────────────

// pickModel selects the best model for the current iteration using this priority:
//  1. Models with headroom > switchThreshold, sorted by largest context window
//  2. If all are below threshold, use the one with the most remaining headroom
//
// Returns ok=false only if zero models have any API key configured.
func pickModel(models []Model, pools map[string]*KeyPool) (Model, *KeyPool, bool) {
	type scored struct {
		model Model
		pool  *KeyPool
		head  float64
	}

	var viable []scored // above threshold
	var best scored
	haveAny := false

	for _, m := range models {
		p := pools[m.Key]
		if p == nil || !p.Available() {
			continue
		}
		h := p.Headroom()
		s := scored{m, p, h}

		if h > switchThreshold {
			viable = append(viable, s)
		}
		if !haveAny || h > best.head {
			best = s
			haveAny = true
		}
	}

	if !haveAny {
		return Model{}, nil, false
	}

	if len(viable) == 0 {
		return best.model, best.pool, true
	}

	chosen := viable[0]
	for _, s := range viable[1:] {
		if s.model.ContextWindow > chosen.model.ContextWindow {
			chosen = s
		}
	}
	return chosen.model, chosen.pool, true
}


// pickModelRoundRobin cycles through available models in order, skipping
// exhausted ones. rrIdx is the current position and is incremented each call.
// Returns ok=false only when every model is exhausted.
func pickModelRoundRobin(models []Model, pools map[string]*KeyPool, rrIdx *int) (Model, *KeyPool, bool) {
	if len(models) == 0 {
		return Model{}, nil, false
	}
	for i := 0; i < len(models); i++ {
		idx := (*rrIdx + i) % len(models)
		m := models[idx]
		p := pools[m.Key]
		if p == nil || !p.Available() || p.Exhausted() {
			continue
		}
		*rrIdx = (idx + 1) % len(models)
		return m, p, true
	}
	return Model{}, nil, false
}

// is429 returns true if an error looks like a quota/rate-limit response.
func is429(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "429") ||
		strings.Contains(msg, "quota") ||
		strings.Contains(msg, "rate limit") ||
		strings.Contains(msg, "RESOURCE_EXHAUSTED")
}

// ── Entry point ───────────────────────────────────────────────────────────────

// RunSelfLoop runs either a task loop (task != "") or the self-improvement loop
// until maxIterations is reached (0 = run forever) or Ctrl-C.
func RunSelfLoop(models []Model, pools map[string]*KeyPool, maxIterations int, task string) {
	_, _, ok := pickModel(models, pools)
	if !ok {
		fmt.Println(Red + "  No models available." + Reset)
		return
	}

	if task != "" {
		RunTaskLoop(task, models, pools, maxIterations)
		return
	}

	fmt.Printf("\n%s%s  Self-improvement mode%s\n", Bold, Yellow, Reset)
	fmt.Printf("%s  Source dir : %s%s\n", Grey, mustCwd(), Reset)
	fmt.Printf("%s  Memory     : %s%s\n", Grey, memoryFile, Reset)
	fmt.Printf("%s  Switch at  : <%.0f%% daily headroom%s\n\n", Grey, switchThreshold*100, Reset)
	fmt.Printf("%s  Ctrl-C to stop.%s\n\n", Yellow, Reset)

	initMemory()

	var currentModelKey string
	iteration := 1

	for {
		if maxIterations > 0 && iteration > maxIterations {
			break
		}

		m, pool, ok := pickModel(models, pools)
		if !ok {
			printSelfError(fmt.Errorf("all models exhausted. Resets at midnight."))
			break
		}

		// Announce model switches
		if currentModelKey != "" && m.Key != currentModelKey {
			col := providerColor(m.Provider)
			fmt.Printf("\n%s%s  ⇄  switching to %s%s\n", col, Bold, m.Name, Reset)
			appendMemory(fmt.Sprintf(
				"**[switched to %s]** Previous model near daily limit.\n\n", m.Name,
			))
		}
		currentModelKey = m.Key

		printSelfHeader(iteration, m, pools)

		err := selfIteration(m, pool, models, pools, iteration)
		if err != nil {
			if is429(err) {
				// Mark this model as heavily used by burning a fake slot, then immediately
				// re-pick — the next iteration will choose a different model.
				printSelfModelWarn(m, fmt.Sprintf("429 from %s — switching model next iteration", m.Name))
				appendMemory(fmt.Sprintf(
					"**[429 on %s]** Hit quota mid-session, will switch.\n\n", m.Name,
				))
				// Don't sleep long — just re-pick
				time.Sleep(2 * time.Second)
			} else {
				printSelfIterationError(iteration, err)
				time.Sleep(10 * time.Second)
			}
		}

		iteration++

		fmt.Printf("%s  — sleeping 5s before next iteration —%s\n\n", Grey, Reset)
		time.Sleep(5 * time.Second)
	}
}


// ── Task loop ────────────────────────────────────────────────────────────────────────────

const taskWorkDir = "task"
const taskMemFile = "task_memory.md"

// spiralThreshold — if the model reply shares this fraction of words with the
// previous reply we consider it stuck and inject a warning.
const spiralThreshold = 0.6

// wordSet returns a simple set of lowercase words from s.
func wordSet(s string) map[string]bool {
	set := map[string]bool{}
	for _, w := range strings.Fields(strings.ToLower(s)) {
		if len(w) > 4 { // ignore short filler words
			set[w] = true
		}
	}
	return set
}

// isSpiralReply returns true when reply looks too similar to lastReply.
func isSpiralReply(reply, lastReply string) bool {
	if lastReply == "" || len(reply) < 50 {
		return false
	}
	cur := wordSet(reply)
	prev := wordSet(lastReply)
	if len(cur) == 0 {
		return false
	}
	overlap := 0
	for w := range cur {
		if prev[w] {
			overlap++
		}
	}
	return float64(overlap)/float64(len(cur)) >= spiralThreshold
}

// RunTaskLoop drives an agentic loop aimed at completing an arbitrary task.
// The agent uses shell, write_file, read_file, fetch_url freely — no file
// allowlist. Each iteration is one Send call that may do many tool calls.
func RunTaskLoop(task string, models []Model, pools map[string]*KeyPool, maxIterations int) {
	fmt.Printf("\n%s%s  Task mode%s\n", Bold, Cyan, Reset)
	fmt.Printf("%s  Task      : %s%s%s\n", Grey, Bold, task, Reset)
	fmt.Printf("%s  Workspace : ./%s/%s\n", Grey, taskWorkDir, Reset)
	fmt.Printf("%s  Memory    : %s%s\n", Grey, taskMemFile, Reset)
	fmt.Printf("%s  Ctrl-C to stop.%s\n\n", Yellow, Reset)

	if err := os.MkdirAll(taskWorkDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "%s  Could not create workspace: %v%s\n", Red, err, Reset)
		return
	}

	initTaskMemory(task)

	var lastReply string
	rrIdx := 0
	spinner := NewSpinner()

	for iteration := 1; maxIterations == 0 || iteration <= maxIterations; iteration++ {
		m, pool, ok := pickModelRoundRobin(models, pools, &rrIdx)
		if !ok {
			printSelfError(fmt.Errorf("all models exhausted. Resets at midnight."))
			break
		}
		col := providerColor(m.Provider)
		fmt.Printf("\n%s%s\u21c4  %s%s\n", col, Bold, m.Name, Reset)

		printSelfHeader(iteration, m, pools)

		memory := readTaskMemory()

		// Spiral detection: inject a stern warning if the model is looping
		spiralWarning := ""
		if isSpiralReply(lastReply, memory) || (lastReply != "" && strings.Contains(lastReply, "Next, I will compile") && strings.Contains(memory, "Next, I will compile")) {
			spiralWarning = "\n\n⚠️  LOOP DETECTED: Your last several replies look nearly identical and you keep saying \"Next, I will compile\" without actually doing it. STOP. Do not rewrite the files again. Instead: (1) run `ls task/` to confirm files exist, (2) run `cd task && gcc *.c -o http_parser 2>&1` to compile, (3) run `./task/http_parser` to test, (4) only fix actual errors shown by the compiler."
			printWarn("Spiral detected — injecting correction prompt")
		}

		history := []Message{
			{Role: "system", Content: taskSystemPrompt(task)},
			{Role: "user", Content: buildTaskPrompt(task, memory, iteration, spiralWarning)},
		}

		spinner.Start(fmt.Sprintf("iteration %d: %s working\u2026", iteration, m.Name))
		start := time.Now()
		reply, err := sendWithLimit(m, pool, history)
		elapsed := time.Since(start)
		spinner.Stop()

		if err != nil {
			if is429(err) {
				printSelfModelWarn(m, fmt.Sprintf("429 from %s \u2014 switching model next iteration", m.Name))
				time.Sleep(2 * time.Second)
				iteration--
				continue
			}
			printSelfIterationError(iteration, err)
			time.Sleep(10 * time.Second)
			continue
		}

		fmt.Printf("%s  replied in %.1fs%s\n", Grey, elapsed.Seconds(), Reset)

		fmt.Printf("\n%s%s\u25b6 %s (iteration %d):%s\n", col, Bold, m.Name, iteration, Reset)
		fmt.Printf("%s%s%s\n\n", Dim, reply, Reset)

		appendTaskMemory(fmt.Sprintf(
			"## Iteration %d \u2014 %s (via %s)\n%s\n\n---\n",
			iteration, time.Now().Format("2006-01-02 15:04"), m.Name, reply,
		))

		lastReply = reply

		if strings.Contains(strings.ToUpper(reply), "TASK COMPLETE") {
			fmt.Printf("%s%s\u2713 Task reported complete. Stopping.%s\n", Green, Bold, Reset)
			break
		}

		fmt.Printf("%s  \u2014 sleeping 5s before next iteration \u2014%s\n\n", Grey, Reset)
		time.Sleep(5 * time.Second)
	}
}

func taskSystemPrompt(task string) string {
	return fmt.Sprintf(`You are an autonomous coding agent. Your task:

  %s

You have tools: shell, write_file, read_file, fetch_url.
All task files live in the ./%s/ subdirectory.

STRICT RULES — follow every one or you will loop forever:
1. ALWAYS compile using full paths from the repo root: gcc task/foo.c task/bar.c -o task/prog
   OR use: cd task && gcc foo.c bar.c -o prog && cd ..
   NEVER run gcc without the task/ prefix or the cd.
2. ALWAYS run the binary after a successful compile and show its actual output.
3. NEVER claim success without showing real compiler output AND real program output.
4. If the previous compile produced errors, READ the error message carefully and fix ONLY those errors.
   Do NOT rewrite working files from scratch — patch the specific broken line.
5. If you have written the same files more than once without compiling, STOP writing and compile NOW.
6. When the task is fully working (binary runs, output is correct), write "TASK COMPLETE" and stop.`, task, taskWorkDir)
}

func buildTaskPrompt(task, memory string, iteration int, spiralWarning string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Task iteration %d\n\n", iteration)
	fmt.Fprintf(&b, "**Task:** %s\n\n", task)
	if memory != "" {
		fmt.Fprintf(&b, "## Progress so far\n\n%s\n\n", memory)
	} else {
		fmt.Fprintf(&b, "## Progress\n\n(first iteration)\n\n")
	}
	if spiralWarning != "" {
		fmt.Fprintf(&b, spiralWarning+"\n\n")
	}
	fmt.Fprintf(&b, "Use your tools. Compile. Run. Show real output. Summarize what you actually did (not what you plan to do).\n")
	return b.String()
}

func initTaskMemory(task string) {
	if _, err := os.Stat(taskMemFile); os.IsNotExist(err) {
		header := fmt.Sprintf("# Task memory\n\n**Task:** %s\n\n", task)
		os.WriteFile(taskMemFile, []byte(header), 0644)
	}
}

func readTaskMemory() string {
	data, err := os.ReadFile(taskMemFile)
	if err != nil {
		return ""
	}
	content := string(data)
	if len(content) > 8000 {
		content = "\u2026(earlier history trimmed)\u2026\n\n" + content[len(content)-8000:]
	}
	return content
}

func appendTaskMemory(entry string) {
	f, err := os.OpenFile(taskMemFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()
	f.WriteString(entry + "\n")
}

// ── Core iteration ────────────────────────────────────────────────────────────

// selfIteration runs one full cycle: read → think → write → build → log.
// models+pools are passed through so selfFix can also switch on 429.
func selfIteration(m Model, pool *KeyPool, models []Model, pools map[string]*KeyPool, iteration int) error {
	sourceContext, err := buildSourceContext()
	if err != nil {
		return fmt.Errorf("reading source: %w", err)
	}
	memory := readMemory()

	history := []Message{
		{Role: "system", Content: selfSystemPrompt()},
		{Role: "user", Content: buildSelfPrompt(sourceContext, memory, iteration)},
	}

	spinner := NewSpinner()
	spinner.Start(fmt.Sprintf("iteration %d: %s thinking…", iteration, m.Name))
	start := time.Now()
	reply, err := sendWithLimit(m, pool, history)
	elapsed := time.Since(start)
	spinner.Stop()

	if err != nil {
		return fmt.Errorf("model error: %w", err)
	}
	fmt.Printf("%s  replied in %.1fs%s\n", Grey, elapsed.Seconds(), Reset)

	plan, err := parseSelfReply(reply)
	if err != nil {
		printSelfWarn(fmt.Sprintf("parse error (logged): %v", err))
		appendMemory(fmt.Sprintf(
			"## Iteration %d — %s\n**Parse error:** %v\n\n**Raw reply (first 500 chars):**\n%.500s\n\n---\n",
			iteration, time.Now().Format("2006-01-02 15:04"), err, reply,
		))
		return nil
	}

	fmt.Printf("\n%s%s  Plan:%s %s\n", Bold, Cyan, Reset, plan.Summary)
	for i, step := range plan.Steps {
		fmt.Printf("  %s%d.%s %s\n", Bold, i+1, Reset, step)
	}
	fmt.Println()

	// Write files
	var changeLog []string
	for _, change := range plan.Changes {
		printSelfInfo(fmt.Sprintf("writing %s", change.File))
		if err := os.WriteFile(change.File, []byte(change.Content), 0644); err != nil {
			msg := fmt.Sprintf("failed to write %s: %v", change.File, err)
			printSelfError(fmt.Errorf(msg))
			changeLog = append(changeLog, "FAILED: "+msg)
			continue
		}
		changeLog = append(changeLog, fmt.Sprintf("wrote %s", change.File))
	}

	// Build
	printSelfInfo("running go build…")
	buildOut, buildErr := runBuild()
	if buildErr != nil {
		printSelfBuildError(buildOut)
		if fixErr := selfFix(models, pools, buildOut); fixErr != nil {
			printSelfError(fmt.Errorf("auto-fix failed: %v", fixErr))
		} else {
			buildOut, buildErr = runBuild()
		}
	} else {
		printSelfBuildOK()
	}

	buildStatus := "✅ build passed"
	if buildErr != nil {
		buildStatus = fmt.Sprintf("❌ build failed:\n%s", buildOut)
	}

	appendMemory(fmt.Sprintf(
		"## Iteration %d — %s (via %s)\n**Summary:** %s\n\n**Changes:**\n%s\n**Build:** %s\n\n**Next focus:** %s\n\n---\n",
		iteration,
		time.Now().Format("2006-01-02 15:04"),
		m.Name,
		plan.Summary,
		bulletList(changeLog),
		buildStatus,
		plan.NextFocus,
	))

	return nil
}

// selfFix picks the best available model (may differ from the one that just failed)
// and tries up to maxFixAttempts times to get a clean build.
func selfFix(models []Model, pools map[string]*KeyPool, initialBuildOutput string) error {
	buildOutput := initialBuildOutput

	for attempt := 1; attempt <= maxFixAttempts; attempt++ {
		m, pool, ok := pickModel(models, pools)
		if !ok {
			return fmt.Errorf("no models available for fix")
		}

		sourceContext, _ := buildSourceContext()
		fixPrompt := fmt.Sprintf(
			"The previous edits caused a build failure (attempt %d/%d).\n\n"+
				"Compiler output:\n```\n%s\n```\n\n"+
				"Current source:\n\n%s\n\n"+
				"Respond with ONLY the JSON object. Fix all errors. "+
				"If needed, revert to a simpler version that compiles.",
			attempt, maxFixAttempts, buildOutput, sourceContext,
		)

		history := []Message{
			{Role: "system", Content: selfSystemPrompt()},
			{Role: "user", Content: fixPrompt},
		}

		spinner := NewSpinner()
		spinner.Start(fmt.Sprintf("fixing via %s (attempt %d/%d)…", m.Name, attempt, maxFixAttempts))
		reply, err := sendWithLimit(m, pool, history)
		spinner.Stop()

		if err != nil {
			printSelfFixModelError(attempt, m, err)
			continue // try next attempt, pickModel may choose differently
		}

		plan, err := parseSelfReply(reply)
		if err != nil {
			printSelfFixError(attempt, err)
			continue
		}

		for _, change := range plan.Changes {
			printSelfInfo(fmt.Sprintf("fix %d: writing %s", attempt, change.File))
			os.WriteFile(change.File, []byte(change.Content), 0644)
		}

		out, buildErr := runBuild()
		if buildErr == nil {
			printSelfFixOK(attempt, m)
			return nil
		}
		buildOutput = out
		printSelfFixBuildError(attempt, out)
	}

	return fmt.Errorf("still broken after %d fix attempts", maxFixAttempts)
}

// ── Prompts ───────────────────────────────────────────────────────────────────

func selfSystemPrompt() string {
	return `You are an autonomous software agent whose job is to iteratively improve your own Go source code.

Rules:
1. You have full access to your source files and can rewrite any of them.
2. After each change the code will be compiled with "go build". If it fails you will be asked to fix it.
3. You MUST respond with a single valid JSON object and nothing else — no markdown fences, no prose before or after.
4. The JSON format is:

{
  "summary": "one-line description of what you are doing this iteration",
  "steps": ["step 1 reasoning", "step 2 reasoning", ...],
  "changes": [
    {"file": "filename.go", "content": "full new file content as a string"},
    ...
  ],
  "next_focus": "what you plan to improve in the next iteration"
}

Guidelines for self-improvement:
- Prefer small, focused changes per iteration. Don't rewrite everything at once.
- Improve error handling, code clarity, performance, or add useful features.
- Keep the program working — a compiling improvement is better than a broken ambitious one.
- You can add new tools, improve the UI, refactor conversation handling, add token counting, etc.
- Track your progress and plans in next_focus so you remember across iterations.
- If a file doesn't need changes, omit it from "changes".
- Write complete file contents — not diffs or partial files.`
}

func buildSelfPrompt(sourceContext, memory string, iteration int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Self-improvement iteration %d\n\n", iteration)
	if memory != "" {
		fmt.Fprintf(&b, "## Memory from previous iterations\n\n%s\n\n", memory)
	} else {
		fmt.Fprintf(&b, "## Memory\n\n(empty — first iteration)\n\n")
	}
	fmt.Fprintf(&b, "## Current source files\n\n%s\n\n", sourceContext)
	fmt.Fprintf(&b, "Analyze the code. Pick ONE meaningful improvement to make this iteration. Output the JSON.\n")
	return b.String()
}

// ── Response parsing ──────────────────────────────────────────────────────────

type fileChange struct {
	File    string `json:"file"`
	Content string `json:"content"`
}

type selfPlan struct {
	Summary   string       `json:"summary"`
	Steps     []string     `json:"steps"`
	Changes   []fileChange `json:"changes"`
	NextFocus string       `json:"next_focus"`
}

func parseSelfReply(raw string) (*selfPlan, error) {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, "```json")
	raw = strings.TrimPrefix(raw, "```")
	raw = strings.TrimSuffix(raw, "```")
	raw = strings.TrimSpace(raw)

	start := strings.Index(raw, "{")
	end := strings.LastIndex(raw, "}")
	if start == -1 || end == -1 || end <= start {
		fmt.Fprintf(os.Stderr, "RAW REPLY (NO JSON OBJECT FOUND):\n%s\n", raw)
		return nil, fmt.Errorf("no JSON object found in reply (len=%d)", len(raw))
	}
	raw = raw[start : end+1]

	var plan selfPlan
	if err := json.Unmarshal([]byte(raw), &plan); err != nil {
		fmt.Fprintf(os.Stderr, "RAW REPLY (JSON PARSE ERROR):\n%s\n", raw)
		return nil, fmt.Errorf("JSON parse error: %w", err)
	}
	if plan.Summary == "" {
		return nil, fmt.Errorf("plan has no summary")
	}

	allowed := map[string]bool{}
	for _, f := range sourceFiles {
		allowed[f] = true
	}
	for _, c := range plan.Changes {
		if !allowed[c.File] {
			return nil, fmt.Errorf("disallowed file: %s", c.File)
		}
		if c.Content == "" {
			return nil, fmt.Errorf("empty content for: %s", c.File)
		}
	}
	return &plan, nil
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func buildSourceContext() (string, error) {
	var b strings.Builder
	for _, name := range sourceFiles {
		data, err := os.ReadFile(name)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return "", err
		}
		fmt.Fprintf(&b, "### %s\n```go\n%s\n```\n\n", name, string(data))
	}
	return b.String(), nil
}

func initMemory() {
	if _, err := os.Stat(memoryFile); os.IsNotExist(err) {
		os.WriteFile(memoryFile, []byte("# koala self-improvement memory\n\n"), 0644)
	}
}

func readMemory() string {
	data, err := os.ReadFile(memoryFile)
	if err != nil {
		return ""
	}
	content := string(data)
	// Keep only the last ~8000 chars — enough for ~6 iterations of context
	if len(content) > 8000 {
		content = "…(earlier history trimmed)…\n\n" + content[len(content)-8000:]
	}
	return content
}

func appendMemory(entry string) {
	f, err := os.OpenFile(memoryFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()
	f.WriteString(entry + "\n")
}

func runBuild() (string, error) {
	result, isErr := toolShell(map[string]interface{}{"command": "go build ./..."})
	if isErr || strings.Contains(result, "[exit:") {
		return result, fmt.Errorf("build failed")
	}
	return result, nil
}

func mustCwd() string {
	cwd, _ := os.Getwd()
	return cwd
}

func bulletList(items []string) string {
	var b strings.Builder
	for _, item := range items {
		fmt.Fprintf(&b, "- %s\n", item)
	}
	return b.String()
}

// ── UI ────────────────────────────────────────────────────────────────────────

func printSelfInfo(msg string) {
	printInfo(msg)
}

func printSelfWarn(msg string) {
	printWarn(msg)
}

func printSelfError(err error) {
	fmt.Printf("%s%s✖ Error:%s %v\n\n", Red, Bold, Reset, err)
}

func printSelfModelWarn(m Model, msg string) {
	col := providerColor(m.Provider)
	fmt.Printf("%s%s⚠ %s (%s):%s %s\n\n", Yellow, Bold, m.Name, col, Reset, msg)
}

func printSelfIterationError(iteration int, err error) {
	fmt.Printf("%s%s✖ Iteration %d error:%s %v\n\n", Red, Bold, iteration, Reset, err)
}

func printSelfBuildError(buildOutput string) {
	fmt.Printf("%s%s✖ Build FAILED:%s\n%s\n", Red+Bold, Reset, buildOutput)
}

func printSelfBuildOK() {
	fmt.Printf("%s%s✓ Build OK%s\n", Green+Bold, Reset)
}

func printSelfFixError(attempt int, err error) {
	fmt.Printf("%s%s✖ Fix attempt %d error:%s %v\n\n", Red, Bold, attempt, Reset, err)
}

func printSelfFixModelError(attempt int, m Model, err error) {
	col := providerColor(m.Provider)
	fmt.Printf("%s%s✖ Fix attempt %d (%s) model error:%s %v\n\n", Red, Bold, attempt, col+m.Name+Reset, Reset, err)
}

func printSelfFixBuildError(attempt int, buildOutput string) {
	fmt.Printf("%s%s✖ Fix attempt %d build FAILED:%s\n%s\n", Red+Bold, Reset, attempt, buildOutput)
}

func printSelfFixOK(attempt int, m Model) {
	col := providerColor(m.Provider)
	fmt.Printf("%s%s✓ Fixed — build OK (attempt %d, via %s)%s\n", Green+Bold, Reset, attempt, col+m.Name+Reset, Reset)
}

func printSelfHeader(iteration int, m Model, pools map[string]*KeyPool) {
	line := strings.Repeat("─", 60)
	col := providerColor(m.Provider)
	fmt.Printf("%s%s%s\n", Grey, line, Reset)
	fmt.Printf("%s%s  Iteration %d%s  %svia %s%s\n",
		Bold, Cyan, iteration, Reset, col+Bold, m.Name, Reset)

	// Headroom indicators — show available/total keys + best headroom
	parts := []string{}
	for _, name := range []string{"gemini-flash", "codestral", "llama-70b", "qwen3-32b"} {
		p, ok := pools[name]
		if !ok {
			continue
		}
		avail, total := 0, len(p.Entries)
		for _, e := range p.Entries {
			if !e.Limiter.Exhausted() {
				avail++
			}
		}
		h := p.Headroom()
		var dot string
		switch {
		case avail == 0:
			dot = Red + "✗" + Reset
		case h > 0.5:
			dot = Green + "●" + Reset
		case h > switchThreshold:
			dot = Yellow + "◐" + Reset
		default:
			dot = Red + "○" + Reset
		}
		var label string
		if total > 1 {
			label = fmt.Sprintf("%d/%d keys·%.0f%%", avail, total, h*100)
		} else {
			label = fmt.Sprintf("%.0f%%", h*100)
		}
		parts = append(parts, fmt.Sprintf("%s %s%s%s", dot, Grey, label, Reset))
	}
	if len(parts) > 0 {
		fmt.Printf("%s  quota: %s%s\n", Grey, strings.Join(parts, "  "), Reset)
	}
	fmt.Printf("%s%s%s\n\n", Grey, line, Reset)
}
