package deepq

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/gammazero/deque"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"gorgonia.org/tensor"
)

// Event is an event that occurred.
type Event struct {
	*envv1.Outcome
	State  *tensor.Dense
	Action int
}

// NewEvent returns a new event
func NewEvent(state *tensor.Dense, action int, outcome *envv1.Outcome) *Event {
	return &Event{
		State:   state,
		Action:  action,
		Outcome: outcome,
	}
}

// Memory for the dqn agent.
type Memory struct {
	*deque.Deque
}

// NewMemory returns a new Memory store.
func NewMemory() *Memory {
	return &Memory{
		Deque: &deque.Deque{},
	}
}

// Sample from the memory with the given batch size.
func (m *Memory) Sample(batchsize int) ([]*Event, error) {
	if m.Len() < batchsize {
		return nil, fmt.Errorf("queue size %d is less than batch size %d", m.Len(), batchsize)
	}
	events := []*Event{}
	rand.Seed(time.Now().UnixNano())
	for i, value := range rand.Perm(m.Len()) {
		if i >= batchsize {
			break
		}
		event := m.At(value).(*Event)
		events = append(events, event)
	}
	return events, nil
}
