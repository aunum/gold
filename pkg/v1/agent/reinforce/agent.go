// Package reinforce is an agent implementation of the REINFORCE algorithm.
package reinforce

import (
	"fmt"
	"time"

	"golang.org/x/exp/rand"

	"github.com/aunum/gold/pkg/v1/dense"
	"github.com/aunum/goro/pkg/v1/model"
	"gonum.org/v1/gonum/stat/distuv"

	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	"github.com/aunum/gold/pkg/v1/common/num"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	"gorgonia.org/tensor"
)

// Agent is a dqn agent.
type Agent struct {
	// Base for the agent.
	*agentv1.Base

	// Hyperparameters for the dqn agent.
	*Hyperparameters

	// Policy by which the agent acts.
	Policy model.Model

	// Memory of the agent.
	Memory *Memory

	env *envv1.Env
}

// Hyperparameters for the dqn agent.
type Hyperparameters struct {
	// Gamma is the discount factor (0≤γ≤1). It determines how much importance we want to give to future
	// rewards. A high value for the discount factor (close to 1) captures the long-term effective award, whereas,
	// a discount factor of 0 makes our agent consider only immediate reward, hence making it greedy.
	Gamma float32
}

// DefaultHyperparameters are the default hyperparameters.
var DefaultHyperparameters = &Hyperparameters{
	Gamma: 0.99,
}

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
	Base:            agentv1.NewBase("REINFORCE"),
}

// NewAgent returns a new dqn agent.
func NewAgent(c *AgentConfig, env *envv1.Env) (*Agent, error) {
	if c == nil {
		c = DefaultAgentConfig
	}
	if c.Base == nil {
		c.Base = DefaultAgentConfig.Base
	}
	if env == nil {
		return nil, fmt.Errorf("environment cannot be nil")
	}
	policy, err := MakePolicy(c.PolicyConfig, c.Base, env)
	if err != nil {
		return nil, err
	}
	return &Agent{
		Base:            c.Base,
		Hyperparameters: c.Hyperparameters,
		Memory:          NewMemory(),
		Policy:          policy,
		env:             env,
	}, nil
}

// Learn the agent.
func (a *Agent) Learn() error {
	states, actions, rewards := a.Memory.Pop()
	err := a.Policy.ResizeBatch(len(states))
	if err != nil {
		return err
	}

	// discount future rewards
	discounted := make([]float32, len(rewards))
	var running float32
	for i := len(rewards) - 1; i >= 0; i-- {
		running = rewards[i] + a.Gamma*running
		discounted[i] = running
	}

	// normalize rewards
	rewardsT := tensor.New(tensor.WithBacking(discounted))
	rewardsNorm, err := dense.ZNorm(rewardsT)
	if err != nil {
		return err
	}

	// make advantage
	advShape := []int{len(states)}
	advShape = append(advShape, a.Policy.Y().Squeeze()...)
	advantages := dense.Zeros(tensor.Float32, advShape...)
	for i := 0; i < len(states); i++ {
		advantages.SetAt(rewardsNorm.Get(i), i, int(actions[i]))
	}

	statesT, err := dense.Concat(0, states...)
	if err != nil {
		return err
	}

	err = a.Policy.FitBatch(statesT, advantages)
	if err != nil {
		return err
	}
	return nil
}

// Action selects the best known action for the given state.
func (a *Agent) Action(state *tensor.Dense) (action int, err error) {
	actionProbsVal, err := a.Policy.Predict(state)
	if err != nil {
		return action, err
	}
	actionProbs := actionProbsVal.(*tensor.Dense)

	// Get action as a random value of the probability distribution.
	weights := num.F32SliceToF64(actionProbs.Data().([]float32))
	dist := distuv.NewCategorical(weights, rand.NewSource(uint64(time.Now().UnixNano())))
	action = int(dist.Rand())
	return
}
