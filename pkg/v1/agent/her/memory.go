package her

import (
	"fmt"
	"math/rand"
	"time"

	envv1 "github.com/aunum/gold/pkg/v1/env"
	"github.com/aunum/log"
	"gorgonia.org/tensor"
)

// Event is an event that occurred.
type Event struct {
	*envv1.Outcome

	// State by which the action was taken.
	State *tensor.Dense

	// Goal the agent is trying to reach.
	Goal *tensor.Dense

	i int
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

// Print the event.
func (e *Event) Print() {
	log.Break()
	log.Infov("state", e.State.Data())
	log.Infov("goal", e.Goal.Data())
	log.Infov("action", e.Outcome.Action)
	log.Infov("reward", e.Outcome.Reward)
	log.Infov("observation", e.Outcome.Observation.Data())
	log.Infov("done", e.Outcome.Done)
	log.Infov("index", e.i)
	log.Break()
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
	size   int
}

// NewMemory returns a new Memory store.
func NewMemory(size int) *Memory {
	return &Memory{
		events: []*Event{},
		size:   size,
	}
}

// Remember events.
func (m *Memory) Remember(events ...*Event) {
	m.events = append(m.events, events...)
	if len(m.events) > m.size {
		m.events = m.events[len(events):]
	}
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
		event.i = value
		events = append(events, event)
	}
	return events, nil
}

// Len of the memory.
func (m *Memory) Len() int {
	return len(m.events)
}
