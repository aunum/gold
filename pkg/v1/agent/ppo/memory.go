package ppo

import (
	"github.com/aunum/gold/pkg/v1/common/num"
	"github.com/aunum/gold/pkg/v1/dense"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	"gorgonia.org/tensor"
)

// Event is an event that occurred when interacting with an environment.
type Event struct {
	State, ActionProbs, ActionOneHot, QValue, Mask, Reward *tensor.Dense
}

// NewEvent returns a new event.
func NewEvent(state, actionProbs, actionOneHot, qValue *tensor.Dense) *Event {
	return &Event{
		State:        state,
		ActionProbs:  actionProbs,
		ActionOneHot: actionOneHot,
		QValue:       qValue,
	}
}

// Apply an outcome to an event.
func (e *Event) Apply(outcome *envv1.Outcome) {
	mask := float32(num.BoolToInt(!outcome.Done))
	e.Mask = tensor.New(tensor.WithBacking([]float32{mask}))
	e.Reward = tensor.New(tensor.WithBacking([]float32{outcome.Reward}))
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
	m.events.ActionProbs = append(m.events.ActionProbs, event.ActionProbs)
	m.events.ActionOneHots = append(m.events.ActionOneHots, event.ActionOneHot)
	m.events.QValues = append(m.events.QValues, event.QValue)
	m.events.Masks = append(m.events.Masks, event.Mask)
	m.events.Rewards = append(m.events.Rewards, event.Reward)
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
	States, ActionProbs, ActionOneHots, QValues, Masks, Rewards []*tensor.Dense
}

// BatchedEvents are the events as a batched tensor.
type BatchedEvents struct {
	States, ActionProbs, ActionOneHots, QValues, Masks, Rewards *tensor.Dense
	Len                                                         int
}

// Batch the events.
func (e *Events) Batch() (events *BatchedEvents, err error) {
	states, err := dense.Concat(0, e.States...)
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
	qValues, err := dense.Concat(0, e.QValues...)
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
	return &BatchedEvents{
		States:        states,
		ActionProbs:   actionProbs,
		ActionOneHots: actionOneHots,
		QValues:       qValues,
		Masks:         masks,
		Rewards:       rewards,
	}, nil

}
