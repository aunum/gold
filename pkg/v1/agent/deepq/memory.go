package deepq

import (
	"fmt"
	"math/rand"

	"github.com/gammazero/deque"
	"gorgonia.org/tensor"
)

// Event to remember.
type Event struct {
	State     *tensor.Dense
	Action    int
	Reward    int
	NextState *tensor.Dense
	Done      bool
}

// NewEvent is a new event.
func NewEvent(state *tensor.Dense, action int, reward int, nextState *tensor.Dense, done bool) *Event {
	return &Event{
		State:     state,
		Action:    action,
		Reward:    reward,
		NextState: nextState,
		Done:      done,
	}
}

// Memory for the dqn agent.
type Memory struct {
	deque.Deque
}

// NewMemory returns a new Memory store.
func NewMemory() *Memory {
	return &Memory{
		Deque: deque.Deque{},
	}
}

// Sample from the memory with the given batch size.
func (m *Memory) Sample(batchsize int) ([]*Event, error) {
	if m.Len() < batchsize {
		return nil, fmt.Errorf("queue size %d is less than batch size %d", m.Len(), batchsize)
	}
	events := []*Event{}
	for i, value := range rand.Perm(m.Len()) {
		if i >= batchsize {
			break
		}
		event := m.At(value).(*Event)
		events = append(events, event)
	}
	return events, nil
}
