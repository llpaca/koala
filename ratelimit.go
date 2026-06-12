package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"
)

// RateLimiter enforces RPM (requests per minute) and RPD (requests per day)
// for a single model. It blocks until a slot is available.
type RateLimiter struct {
	mu sync.Mutex

	key string // unique key for persistence (e.g. "google-1", "codestral")

	rpm int // max requests per minute (0 = unlimited)
	rpd int // max requests per day   (0 = unlimited)

	// sliding window for per-minute tracking
	minuteWindow []time.Time

	// daily counter (resets at midnight local time)
	dayCount   int
	dayResetAt time.Time
}

// persistedState is the on-disk shape for a single limiter's state.
type persistedState struct {
	DayCount   int       `json:"day_count"`
	DayResetAt time.Time `json:"day_reset_at"`
}

const stateFile = ".koala_state.json"

var (
	stateMu    sync.Mutex
	stateCache map[string]persistedState
)

// loadStateFile reads the persisted state file once (cached).
func loadStateFile() map[string]persistedState {
	stateMu.Lock()
	defer stateMu.Unlock()
	if stateCache != nil {
		return stateCache
	}
	stateCache = map[string]persistedState{}
	data, err := os.ReadFile(stateFile)
	if err != nil {
		return stateCache
	}
	if err := json.Unmarshal(data, &stateCache); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not unmarshal ratelimit state: %v\n", err)
	}
	return stateCache
}

// saveStateFile writes the full state map back to disk.
func saveStateFile() {
	stateMu.Lock()
	defer stateMu.Unlock()
	data, err := json.MarshalIndent(stateCache, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not marshal ratelimit state: %v\n", err)
		return
	}
	if err := os.WriteFile(stateFile, data, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not write ratelimit state file: %v\n", err)
	}
}

// NewRateLimiter creates a rate limiter for the given key, restoring
// persisted daily-usage state from disk if present and still valid.
func NewRateLimiter(key string, rpm, rpd int) *RateLimiter {
	now := time.Now()
	midnight := time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, now.Location())

	rl := &RateLimiter{
		key:		key,
		rpm:		rpm,
		rpd:		rpd,
		dayResetAt: midnight,
	}

	saved := loadStateFile()
	if s, ok := saved[key]; ok {
		if now.Before(s.DayResetAt) {
			rl.dayCount = s.DayCount
			rl.dayResetAt = s.DayResetAt
		}
		// else: stale entry from a previous day, keep fresh midnight reset
	}

	return rl
}

// persist writes this limiter's current daily counter to the shared state file.
func (r *RateLimiter) persist() {
	stateMu.Lock()
	if stateCache == nil {
		stateCache = map[string]persistedState{}
	}
	stateCache[r.key] = persistedState{
		DayCount:   r.dayCount,
		DayResetAt: r.dayResetAt,
	}
	stateMu.Unlock()
	saveStateFile()
}

// tickDay resets the daily counter if midnight has passed. Must be called under mu.
func (r *RateLimiter) tickDay(now time.Time) {
	if now.After(r.dayResetAt) {
		r.dayCount = 0
		for now.After(r.dayResetAt) {
			r.dayResetAt = r.dayResetAt.Add(24 * time.Hour)
		}
		r.persist()
	}
}

// Wait blocks until a request slot is available, then claims it.
// Returns an error if the daily limit is exhausted.
func (r *RateLimiter) Wait() error {
	for {
		r.mu.Lock()
		now := time.Now()
		r.tickDay(now)

		// Check daily cap
		if r.rpd > 0 && r.dayCount >= r.rpd {
			resetAt := r.dayResetAt
			r.mu.Unlock()
			return fmt.Errorf("daily limit of %d requests reached; resets at %s",
				r.rpd, resetAt.Format("15:04"))
		}

		// Prune minute window (keep only last 60s)
		if r.rpm > 0 {
			cutoff := now.Add(-time.Minute)
			keep := r.minuteWindow[:0]
			for _, t := range r.minuteWindow {
				if t.After(cutoff) {
					keep = append(keep, t)
				}
			}
			r.minuteWindow = keep

			if len(r.minuteWindow) >= r.rpm {
				oldest := r.minuteWindow[0]
				waitUntil := oldest.Add(time.Minute)
				r.mu.Unlock()
				time.Sleep(time.Until(waitUntil) + 50*time.Millisecond)
				continue
			}
		}

		// Claim the slot
		r.minuteWindow = append(r.minuteWindow, now)
		r.dayCount++
		r.persist()
		r.mu.Unlock()
		return nil
	}
}

// Headroom returns a 0.0–1.0 score of remaining daily quota.
// 1.0 = fully fresh, 0.0 = exhausted.f
// Models with no daily limit always return 1.0.
func (r *RateLimiter) Headroom() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tickDay(time.Now())

	if r.rpd == 0 {
		return 1.0 // unlimited
	}
	if r.dayCount >= r.rpd {
		return 0.0
	}
	return 1.0 - float64(r.dayCount)/float64(r.rpd)
}

// MarkExhausted forces the daily limit to be considered used up immediately
// (e.g. when the API itself returns a quota-exceeded error, even if our
// local counter hadn't reached rpd yet). Persists so it survives restarts.
func (r *RateLimiter) MarkExhausted() {
	r.mu.Lock()
	defer r.mu.Unlock()
	now := time.Now()
	r.tickDay(now)
	if r.rpd == 0 {
		r.rpd = r.dayCount + 1 // give it a cap so Exhausted() can trigger
	}
	r.dayCount = r.rpd
	r.persist()
}

// DayUsage returns the current daily count and the configured daily limit (0 = unlimited).
func (r *RateLimiter) DayUsage() (used, limit int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tickDay(time.Now())
	return r.dayCount, r.rpd
}

func (r *RateLimiter) Exhausted() bool {
	return r.Headroom() == 0.0
}

// Status returns a human-readable string of current usage.
func (r *RateLimiter) Status() string {
	r.mu.Lock()
	defer r.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-time.Minute)
	recent := 0
	for _, t := range r.minuteWindow {
		if t.After(cutoff) {
			recent++
		}
	}

	if r.rpm == 0 && r.rpd == 0 {
		return "unlimited"
	}
	if r.rpd == 0 {
		return fmt.Sprintf("%d/%d rpm", recent, r.rpm)
	}
	return fmt.Sprintf("%d/%d rpm · %d/%d rpd", recent, r.rpm, r.dayCount, r.rpd)
}
