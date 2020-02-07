package ppo1

import (
	"github.com/pbarker/go-rl/pkg/v1/dense"
	"gorgonia.org/tensor"
)

// Event is an event that occurred.
type Event struct {
	// State by which the action was taken.
	State, Action, Value, Mask, Reward, ActionProbs, ActionOneHot *tensor.Dense
}

// NewEvent returns a new event
func NewEvent(state, action, value, mask, reward, actionProbs, actionOneHot *tensor.Dense) *Event {
	return &Event{state, action, value, mask, reward, actionProbs, actionOneHot}
}

// Memory for the dqn agent.
type Memory struct {
	events *Events
}

// NewMemory returns a new Memory store.
func NewMemory() *Memory {
	return &Memory{}
}

// Remember an event.
func (m *Memory) Remember(event *Event) error {
	m.events.States = append(m.events.States, event.State)
	m.events.Actions = append(m.events.Actions, event.Action)
	m.events.Values = append(m.events.Values, event.Value)
	m.events.Masks = append(m.events.Masks, event.Mask)
	m.events.Rewards = append(m.events.Rewards, event.Reward)
	m.events.ActionProbs = append(m.events.ActionProbs, event.ActionProbs)
	m.events.ActionOneHots = append(m.events.ActionOneHots, event.ActionOneHot)
	return nil
}

// Pop the values out of the memory.
func (m *Memory) Pop() (e *Events) {
	e = m.events
	m.Reset()
	return
}

// Reset the memory.
func (m *Memory) Reset() {
	m.events = nil
}

// Len is the number of events in the memory.
func (m *Memory) Len() int {
	if m.events == nil {
		return 0
	}
	return len(m.events.States)
}

// Events are the events as a batched tensor.
type Events struct {
	States        []*tensor.Dense
	Actions       []*tensor.Dense
	Values        []*tensor.Dense
	Masks         []*tensor.Dense
	Rewards       []*tensor.Dense
	ActionProbs   []*tensor.Dense
	ActionOneHots []*tensor.Dense
}

// BatchedEvents are the events as a batched tensor.
type BatchedEvents struct {
	States        *tensor.Dense
	Actions       *tensor.Dense
	Values        *tensor.Dense
	Masks         *tensor.Dense
	Rewards       *tensor.Dense
	ActionProbs   *tensor.Dense
	ActionOneHots *tensor.Dense
	Len           int
}

// Batch the events.
func (e *Events) Batch(event *Event) (events *BatchedEvents, err error) {
	states, err := dense.Concat(0, e.States...)
	if err != nil {
		return nil, err
	}
	actions, err := dense.Concat(0, e.Actions...)
	if err != nil {
		return nil, err
	}
	values, err := dense.Concat(0, e.Values...)
	if err != nil {
		return nil, err
	}
	masks, err := dense.Concat(0, e.Masks...)
	if err != nil {
		return nil, err
	}
	rewards, err := dense.Concat(0, e.Rewards...)
	if err != nil {
		return nil, err
	}
	actionProbs, err := dense.Concat(0, e.ActionProbs...)
	if err != nil {
		return nil, err
	}
	actionOneHots, err := dense.Concat(0, e.ActionOneHots...)
	if err != nil {
		return nil, err
	}
	return &BatchedEvents{
		States:        states,
		Actions:       actions,
		Values:        values,
		Masks:         masks,
		Rewards:       rewards,
		ActionProbs:   actionProbs,
		ActionOneHots: actionOneHots,
	}, nil

}
