package main

import "sync"

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Conversation holds per-model chat history.
type Conversation struct {
	mu       sync.RWMutex
	messages []Message
}

func NewConversation() *Conversation {
	return &Conversation{}
}

func (c *Conversation) Add(role, content string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = append(c.messages, Message{Role: role, Content: content})
}

func (c *Conversation) Get() []Message {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]Message, len(c.messages))
	copy(out, c.messages)
	return out
}

func (c *Conversation) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.messages = nil
}

func (c *Conversation) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.messages)
}
