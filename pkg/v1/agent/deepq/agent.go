package deepq

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/pbarker/go-rl/pkg/v1/common"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"gorgonia.org/tensor"
)

// Agent is a dqn agent.
type Agent struct {
	// Hyperparameters for the dqn agent.
	*Hyperparameters

	policy *Policy
	env    *envv1.Env

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
	// TODO: these should be a schedule of some sort.
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
func NewAgent(c *AgentConfig, env *envv1.Env) (*Agent, error) {
	policy, err := NewPolicy(c.PolicyConfig, env)
	if err != nil {
		return nil, err
	}
	return &Agent{
		Hyperparameters: c.Hyperparameters,
		epsilon:         c.EpsilonMax,
		memory:          NewMemory(),
		policy:          policy,
		env:             env,
	}, nil
}

// Learn the agent.
func (a *Agent) Learn() error {
	if a.memory.Len() < a.ReplayBatchSize {
		return nil
	}
	batch, err := a.memory.Sample(a.ReplayBatchSize)
	if err != nil {
		return err
	}
	for _, event := range batch {
		update := float32(event.Reward)
		if !event.Done {
			nextAction, err := a.action(event.NextState)
			if err != nil {
				return err
			}
			update = (float32(event.Reward) + a.Gamma*float32(nextAction))
		}
		qValues, err := a.policy.Predict(event.State)
		if err != nil {
			return err
		}
		qValues.Set(event.Action, update)
		err = a.policy.Fit(event.State, qValues)
		if err != nil {
			return err
		}
	}
	a.epsilon *= a.EpsilonDecay
	a.epsilon = math32.Max(a.EpsilonMin, a.epsilon)
	return nil
}

// Action selects the best known action for the given state.
func (a *Agent) Action(state *tensor.Dense) (action int, err error) {
	if common.RandFloat32(float32(0.0), float32(1.0)) < a.epsilon {
		// explore
		action, err = a.env.SampleAction()
		if err != nil {
			return
		}
	}
	action, err = a.action(state)
	return
}

func (a *Agent) action(state *tensor.Dense) (action int, err error) {
	qValues, err := a.policy.Predict(state)
	if err != nil {
		return
	}
	fmt.Println("qvalues: ", qValues)
	actionIndex, err := qValues.Argmax(0)
	if err != nil {
		return action, err
	}
	action = actionIndex.GetI(0)
	fmt.Println("action: ", action)
	return
}

// Remember an event.
func (a *Agent) Remember(event *Event) {
	a.memory.PushFront(event)
}
