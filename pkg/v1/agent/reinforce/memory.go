package reinforce

import (
	"gorgonia.org/tensor"
)

// Memory for the agent.
type Memory struct {
	States  []*tensor.Dense
	Actions []float32
	Rewards []float32
}

// NewMemory returns a new Memory store.
func NewMemory() *Memory {
	return &Memory{
		Actions: []float32{},
		Rewards: []float32{},
	}
}

// Clear the memory.
func (m *Memory) Clear() {
	m.States = []*tensor.Dense{}
	m.Actions = []float32{}
	m.Rewards = []float32{}
}

// Store an action reward pair.
func (m *Memory) Store(state *tensor.Dense, action int, reward float32) {
	m.States = append(m.States, state)
	m.Actions = append(m.Actions, float32(action))
	m.Rewards = append(m.Rewards, reward)
}

// Pop the actions and rewards from memory.
func (m *Memory) Pop() (states []*tensor.Dense, actions, rewards []float32) {
	states = m.States
	actions = m.Actions
	rewards = m.Rewards
	m.Clear()
	return
}
