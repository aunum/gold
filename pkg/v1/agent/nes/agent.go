package nes

import (
	"fmt"

	"github.com/pbarker/go-rl/pkg/v1/model"

	agentv1 "github.com/pbarker/go-rl/pkg/v1/agent"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"gorgonia.org/tensor"
)

// Agent is a dqn agent.
type Agent struct {
	// Base for the agent.
	*agentv1.Base

	// Hyperparameters for the dqn agent.
	*Hyperparameters

	Policy model.Model
	env    *envv1.Env
}

// Hyperparameters for the dqn agent.
type Hyperparameters struct {
	// Gamma is the discount factor (0≤γ≤1). It determines how much importance we want to give to future
	// rewards. A high value for the discount factor (close to 1) captures the long-term effective award, whereas,
	// a discount factor of 0 makes our agent consider only immediate reward, hence making it greedy.
	Gamma float32
}

// DefaultHyperparameters are the default hyperparameters.
var DefaultHyperparameters = &Hyperparameters{}

// AgentConfig is the config for a dqn agent.
type AgentConfig struct {
	// Base for the agent.
	Base *agentv1.Base

	// Hyperparameters for the agent.
	*Hyperparameters

	// PolicyConfig for the agent.
	PolicyConfig *PolicyConfig
}

// DefaultAgentConfig is the default config for a dqn agent.
var DefaultAgentConfig = &AgentConfig{
	Hyperparameters: DefaultHyperparameters,
	PolicyConfig:    DefaultPolicyConfig,
	Base:            agentv1.NewBase(agentv1.WithNoTracker()),
}

// NewAgent returns a new dqn agent.
func NewAgent(c *AgentConfig, env *envv1.Env) (*Agent, error) {
	if c == nil {
		c = DefaultAgentConfig
	}
	if c.Base == nil {
		c.Base = agentv1.NewBase()
	}
	if env == nil {
		return nil, fmt.Errorf("environment cannot be nil")
	}
	return &Agent{
		Base:            c.Base,
		Hyperparameters: c.Hyperparameters,
		env:             env,
	}, nil
}

// Action selects the best known action for the given state and weights.
func (a *Agent) Action(state, weights *tensor.Dense) (action int, err error) {

	return
}
