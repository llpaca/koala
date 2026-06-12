package main

import (
	"fmt"
	"os"
)

// KeyEntry pairs an API key with its own rate limiter (persisted by env var name).
type KeyEntry struct {
	EnvVar  string
	APIKey  string
	Limiter *RateLimiter
}

// KeyPool holds the rotating set of keys available for a single model.
type KeyPool struct {
	Model   Model
	Entries []KeyEntry
	idx     int // next entry to try
}

// NewKeyPool builds a KeyPool for a model from its EnvKeys list.
func NewKeyPool(m Model) *KeyPool {
	envVars := m.EnvKeys

	var entries []KeyEntry
	for _, ev := range envVars {
		key := os.Getenv(ev)
		if key == "" {
			continue
		}
		entries = append(entries, KeyEntry{
			EnvVar:  ev,
			APIKey:  key,
			Limiter: NewRateLimiter(m.Key+":"+ev, m.RPM, m.RPD),
		})
	}

	return &KeyPool{Model: m, Entries: entries}
}

// Available reports whether the pool has at least one usable key.
func (p *KeyPool) Available() bool {
	return len(p.Entries) > 0
}

// Next returns the next non-exhausted key entry, rotating round-robin.
// Returns an error if every key in the pool is exhausted.
func (p *KeyPool) Next() (*KeyEntry, error) {
	if len(p.Entries) == 0 {
		return nil, fmt.Errorf("no API keys configured for %s", p.Model.Name)
	}

	for i := 0; i < len(p.Entries); i++ {
		e := &p.Entries[(p.idx+i)%len(p.Entries)]
		if !e.Limiter.Exhausted() {
			p.idx = (p.idx + i + 1) % len(p.Entries)
			return e, nil
		}
	}

	return nil, fmt.Errorf("all %d API key(s) for %s have exhausted their daily quota", len(p.Entries), p.Model.Name)
}

// Headroom returns the best (max) headroom across all keys in the pool.

// Exhausted reports whether every key in the pool is exhausted.
func (p *KeyPool) Exhausted() bool {
	for _, e := range p.Entries {
		if !e.Limiter.Exhausted() {
			return false
		}
	}
	return true
}

func (p *KeyPool) Headroom() float64 {
	best := 0.0
	for _, e := range p.Entries {
		if h := e.Limiter.Headroom(); h > best {
			best = h
		}
	}
	return best
}

// Status returns a human-readable summary, e.g. "2/4 keys available · 0/10 rpm · 120/2000 rpd".
func (p *KeyPool) Status() string {
	if len(p.Entries) == 0 {
		return "no keys"
	}
	if len(p.Entries) == 1 {
		return p.Entries[0].Limiter.Status()
	}

	avail := 0
	totalDay, totalRPD := 0, 0
	for _, e := range p.Entries {
		if !e.Limiter.Exhausted() {
			avail++
		}
		d, r := e.Limiter.DayUsage()
		totalDay += d
		totalRPD += r
	}
	return fmt.Sprintf("%d/%d keys available · %d/%d rpd combined", avail, len(p.Entries), totalDay, totalRPD)
}
