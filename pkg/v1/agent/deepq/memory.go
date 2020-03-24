package deepq

import (
	"fmt"
	"math/rand"
	"time"

	envv1 "github.com/aunum/gold/pkg/v1/env"
	"github.com/aunum/log"
	"github.com/gammazero/deque"
	"gorgonia.org/tensor"
)

// Event is an event that occurred.
type Event struct {
	*envv1.Outcome

	// State by which the action was taken.
	State *tensor.Dense

	// Action that was taken.
	Action int

	i int
}

// NewEvent returns a new event
func NewEvent(state *tensor.Dense, action int, outcome *envv1.Outcome) *Event {
	return &Event{
		State:   state,
		Action:  action,
		Outcome: outcome,
	}
}

// Print the event.
func (e *Event) Print() {
	log.Infof("event --> \n state: %v \n action: %v \n reward: %v \n done: %v \n obv: %v\n\n", e.State, e.Action, e.Reward, e.Done, e.Observation)
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
		event.i = value
		events = append(events, event)
	}
	return events, nil
}
