package main

import (
	"fmt"
	"sync"
	"time"
)

// RateLimiter enforces RPM (requests per minute) and RPD (requests per day)
// for a single model. It blocks until a slot is available.
type RateLimiter struct {
	mu sync.Mutex

	rpm int // max requests per minute (0 = unlimited)
	rpd int // max requests per day   (0 = unlimited)

	// sliding window for per-minute tracking
	minuteWindow []time.Time

	// daily counter (resets at midnight local time)
	dayCount    int
	dayResetAt  time.Time
}

func NewRateLimiter(rpm, rpd int) *RateLimiter {
	now := time.Now()
	midnight := time.Date(now.Year(), now.Month(), now.Day()+1, 0, 0, 0, 0, now.Location())
	return &RateLimiter{
		rpm:        rpm,
		rpd:        rpd,
		dayResetAt: midnight,
	}
}

// Wait blocks until a request slot is available, then claims it.
// Returns an error if the daily limit is exhausted (blocking forever would be bad UX).
func (r *RateLimiter) Wait() error {
	for {
		r.mu.Lock()
		now := time.Now()

		// Reset daily counter if the day rolled over
		if now.After(r.dayResetAt) {
			r.dayCount = 0
			next := r.dayResetAt.Add(24 * time.Hour)
			r.dayResetAt = next
		}

		// Check daily cap
		if r.rpd > 0 && r.dayCount >= r.rpd {
			r.mu.Unlock()
			return fmt.Errorf("daily limit of %d requests reached; resets at %s",
				r.rpd, r.dayResetAt.Format("15:04"))
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
				// Must wait until the oldest request in the window is >60s old
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
		r.mu.Unlock()
		return nil
	}
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
