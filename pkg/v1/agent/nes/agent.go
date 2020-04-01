// Package nes is an agent implementation of the Natural Evolution Strategies algorithm.
package nes

import (
	"fmt"

	"github.com/aunum/goro/pkg/v1/model"

	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Agent is a dqn agent.
type Agent struct {
	// Base for the agent.
	*agentv1.Base

	// Policy for the agent.
	Policy model.Model

	env *envv1.Env
}

// AgentConfig is the config for a dqn agent.
type AgentConfig struct {
	// PolicyConfig for the agent.
	PolicyConfig *PolicyConfig
}

// DefaultAgentConfig is the default config for a dqn agent.
var DefaultAgentConfig = &AgentConfig{
	PolicyConfig: DefaultPolicyConfig,
}

// NewAgent returns a new dqn agent.
func NewAgent(c *AgentConfig, env *envv1.Env, base *agentv1.Base) (*Agent, error) {
	if c == nil {
		c = DefaultAgentConfig
	}
	if base == nil {
		base = agentv1.NewBase("NES")
	}
	if env == nil {
		return nil, fmt.Errorf("environment cannot be nil")
	}
	policy, err := MakePolicy(c.PolicyConfig, base, env)
	if err != nil {
		return nil, err
	}
	return &Agent{
		Base:   base,
		env:    env,
		Policy: policy,
	}, nil
}

// Action selects the best known action for the given state and weights.
func (a *Agent) Action(state *tensor.Dense) (action int, err error) {
	prediction, err := a.Policy.Predict(state)
	if err != nil {
		return action, err
	}
	qValues := prediction.(*tensor.Dense)
	actionIndex, err := qValues.Argmax(1)
	if err != nil {
		return action, err
	}
	action = actionIndex.GetI(0)
	return
}

// SetWeights sets the weights.
// TODO: support multiple weights.
func (a *Agent) SetWeights(weights *tensor.Dense) error {
	for _, learnable := range a.Policy.Learnables() {
		err := g.Let(learnable, weights)
		if err != nil {
			return err
		}
	}
	return nil
}
