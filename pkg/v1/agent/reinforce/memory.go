package reinforce

// Memory for the agent.
type Memory struct {
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
	m.Actions = []float32{}
	m.Rewards = []float32{}
}

// Store an action reward pair.
func (m *Memory) Store(action int, reward float32) {
	m.Actions = append(m.Actions, float32(action))
	m.Rewards = append(m.Rewards, reward)
}

// Pop the actions and rewards from memory.
func (m *Memory) Pop() (actions, rewards []float32) {
	actions = m.Actions
	rewards = m.Rewards
	m.Clear()
	return
}
