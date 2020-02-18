package her

import (
	"fmt"
	"math/rand"
	"time"

	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"gorgonia.org/tensor"
)

// Event is an event that occurred.
type Event struct {
	*envv1.Outcome

	// State by which the action was taken.
	State *tensor.Dense

	// Goal the agent is trying to reach.
	Goal *tensor.Dense
}

// Events that occurred.
type Events []*Event

// NewEvent returns a new event
func NewEvent(state, goal *tensor.Dense, outcome *envv1.Outcome) *Event {
	return &Event{
		Outcome: outcome,
		State:   state,
		Goal:    goal,
	}
}

// Copy the event.
func (e *Event) Copy() *Event {
	return &Event{
		Outcome: &envv1.Outcome{
			Observation: e.Outcome.Observation.Clone().(*tensor.Dense),
			Action:      e.Outcome.Action,
			Reward:      e.Outcome.Reward,
			Done:        e.Outcome.Done,
		},
		State: e.State.Clone().(*tensor.Dense),
		Goal:  e.Goal.Clone().(*tensor.Dense),
	}
}

// Copy the events.
func (e Events) Copy() Events {
	ret := Events{}
	for _, event := range e {
		ret = append(ret, event.Copy())
	}
	return ret
}

// Memory for the dqn agent.
type Memory struct {
	events []*Event
}

// NewMemory returns a new Memory store.
func NewMemory() *Memory {
	return &Memory{
		events: []*Event{},
	}
}

// Remember events.
func (m *Memory) Remember(event ...*Event) {
	m.events = append(m.events, event...)
}

// Sample a batch size from memory.
func (m *Memory) Sample(batchsize int) (ret []*Event, err error) {
	if m.Len() < batchsize {
		return nil, fmt.Errorf("queue size %d is less than batch size %d", m.Len(), batchsize)
	}
	events := []*Event{}
	rand.Seed(time.Now().UnixNano())
	for i, value := range rand.Perm(m.Len()) {
		if i >= batchsize {
			break
		}
		event := m.events[value]
		events = append(events, event)
	}
	return events, nil
}

// Len of the memory.
func (m *Memory) Len() int {
	return len(m.events)
}
