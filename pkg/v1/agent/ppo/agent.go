// Package ppo is an agent implementation of the Proximal Policy Optimization algorithm.
package ppo

import (
	"fmt"
	"time"

	"github.com/aunum/gold/pkg/v1/dense"

	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	"github.com/aunum/gold/pkg/v1/common/num"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	modelv1 "github.com/aunum/goro/pkg/v1/model"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
	"gorgonia.org/tensor"
)

// Agent is a dqn agent.
type Agent struct {
	// Base for the agent.
	*agentv1.Base

	// Hyperparameters for the dqn agent.
	*Hyperparameters

	// Actor chooses actions.
	Actor modelv1.Model

	// Critic updates params.
	Critic modelv1.Model

	// Memory of the agent.
	Memory *Memory

	env     *envv1.Env
	epsilon float32

	steps    int
	ppoSteps int
}

// Hyperparameters for the dqn agent.
type Hyperparameters struct {
	// Gamma is the discount factor (0≤γ≤1). It determines how much importance we want to give to future
	// rewards. A high value for the discount factor (close to 1) captures the long-term effective award, whereas,
	// a discount factor of 0 makes our agent consider only immediate reward, hence making it greedy.
	Gamma float32

	// Lambda is the smoothing factor which is used to reduce variance and stablilize training.
	Lambda float32
}

// DefaultHyperparameters are the default hyperparameters.
var DefaultHyperparameters = &Hyperparameters{
	Gamma:  0.99,
	Lambda: 0.95,
}

// AgentConfig is the config for a dqn agent.
type AgentConfig struct {
	// Base for the agent.
	Base *agentv1.Base

	// Hyperparameters for the agent.
	*Hyperparameters

	// ActorConfig is the actor model config.
	ActorConfig *ModelConfig

	// CriticConfig is the critic model config.
	CriticConfig *ModelConfig
}

// DefaultAgentConfig is the default config for a dqn agent.
var DefaultAgentConfig = &AgentConfig{
	Hyperparameters: DefaultHyperparameters,
	Base:            agentv1.NewBase("PPO"),
	ActorConfig:     DefaultActorConfig,
	CriticConfig:    DefaultCriticConfig,
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
	actor, err := MakeActor(c.ActorConfig, c.Base, env)
	if err != nil {
		return nil, err
	}
	critic, err := MakeCritic(c.CriticConfig, c.Base, env)
	if err != nil {
		return nil, err
	}
	return &Agent{
		Base:            c.Base,
		Hyperparameters: c.Hyperparameters,
		Actor:           actor,
		Critic:          critic,
		env:             env,
	}, nil
}

// Learn the agent.
func (a *Agent) Learn(event *Event) error {
	err := a.Memory.Remember(event)
	if err != nil {
		return err
	}
	if a.ppoSteps > a.Memory.Len() {
		return nil
	}
	events := a.Memory.Pop()

	eventsBatch, err := events.Batch()
	if err != nil {
		return err
	}

	// Need one extra qValue for the GAE formula.
	qValue, err := a.Critic.Predict(event.State)
	if err != nil {
		return err
	}
	events.QValues = append(events.QValues, qValue.(*tensor.Dense))

	// calculate the advantages.
	returns, advantages, err := GAE(events.QValues, events.Masks, events.Rewards, a.Gamma, a.Lambda)
	if err != nil {
		return err
	}

	err = a.Actor.Fit(modelv1.Values{
		eventsBatch.States,
		eventsBatch.ActionProbs,
		advantages,
		eventsBatch.Rewards,
		eventsBatch.QValues,
	}, eventsBatch.ActionOneHots)
	if err != nil {
		return err
	}

	err = a.Critic.Fit(eventsBatch.States, returns)
	if err != nil {
		return err
	}

	return nil
}

// Action selects the best known action for the given state.
func (a *Agent) Action(state *tensor.Dense) (action int, event *Event, err error) {
	a.steps++

	actionProbsVal, err := a.Actor.Predict(state)
	if err != nil {
		return action, event, err
	}
	actionProbs := actionProbsVal.(*tensor.Dense)

	// Get action as a random value of the probability distribution.
	weights := num.F32SliceToF64(actionProbs.Data().([]float32))
	dist := distuv.NewCategorical(weights, rand.NewSource(uint64(time.Now().UnixNano())))
	action = int(dist.Rand())

	qv, err := a.Critic.Predict(state)
	if err != nil {
		return action, event, err
	}
	qValue := qv.(*tensor.Dense)

	actionOneHot, err := dense.OneHotVector(action, actionProbs.Shape()[0], tensor.Float32)
	if err != nil {
		return action, event, err
	}

	event = NewEvent(state, actionProbs, actionOneHot, qValue)
	return
}
