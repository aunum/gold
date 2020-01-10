package deepq

import (
	"gorgonia.org/tensor"
)

// Agent is a dqn agent.
type Agent struct {
	// Hyperparameters for the dqn agent.
	*Hyperparameters

	policy *Policy

	epsilon float32
	memory  *Memory
}

// Hyperparameters for the dqn agent.
type Hyperparameters struct {
	// Gamma is the discount factor (0≤γ≤1). It determines how much importance we want to give to future
	// rewards. A high value for the discount factor (close to 1) captures the long-term effective award, whereas,
	// a discount factor of 0 makes our agent consider only immediate reward, hence making it greedy.
	Gamma float32

	// Alpha is the learning rate (0<α≤1). Just like in supervised learning settings, alpha is the extent
	// to which our Q-values are being updated in every iteration.
	Alpha float32

	// EpsilonMin is the minimum rate at which the agent can explore.
	EpsilonMin float32

	// EpsilonMax is the maximum rate at which an agent can explore.
	EpsilonMax float32

	// EpsilonDecay is the rate at which the agent should exploit over explore.
	EpsilonDecay float32

	// ReplayBatchSize determines how large a batch is replayed from memory.
	ReplayBatchSize int
}

// AgentConfig is the config for a dqn agent.
type AgentConfig struct {
	*Hyperparameters
	PolicyConfig *PolicyConfig
}

// DefaultAgentConfig is the default config for a dqn agent.
var DefaultAgentConfig = &AgentConfig{
	Hyperparameters: &Hyperparameters{
		Gamma:        0.95,
		Alpha:        0.001,
		EpsilonMin:   0.01,
		EpsilonMax:   1.0,
		EpsilonDecay: 0.995,
	},
	PolicyConfig: DefaultPolicyConfig,
}

// NewAgent returns a new dqn agent.
func NewAgent(c *AgentConfig, actionSpaceSize int) (*Agent, error) {
	policy, err := NewPolicy(c.PolicyConfig, actionSpaceSize)
	if err != nil {
		return nil, err
	}
	return &Agent{
		Hyperparameters: c.Hyperparameters,
		epsilon:         c.EpsilonMax,
		memory:          NewMemory(),
		policy:          policy,
	}, nil
}

// Learn the agent.
func (a *Agent) Learn() {

}

// Action has the agent pick an action for the given state.
func (a *Agent) Action(state *tensor.Dense) (int, error) {
	return 0, nil
}
