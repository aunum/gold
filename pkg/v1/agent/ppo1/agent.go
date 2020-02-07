package ppo1

import (
	"fmt"

	agentv1 "github.com/pbarker/go-rl/pkg/v1/agent"
	"github.com/pbarker/go-rl/pkg/v1/common"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	modelv1 "github.com/pbarker/go-rl/pkg/v1/model"
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

	Epsilon common.Schedule
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

	// Epsilon is the rate at which the agent should exploit vs explore.
	Epsilon common.Schedule
}

// DefaultHyperparameters are the default hyperparameters.
var DefaultHyperparameters = &Hyperparameters{
	Epsilon: common.DefaultDecaySchedule(),
	Gamma:   0.95,
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
	Base:            agentv1.NewBase(),
	ActorConfig:     DefaultActorConfig,
	CriticConfig:    DefaultCriticConfig,
}

// NewAgent returns a new dqn agent.
func NewAgent(c *AgentConfig, env *envv1.Env) (*Agent, error) {
	if c == nil {
		c = DefaultAgentConfig
	}
	if c.Base == nil {
		c.Base = agentv1.NewBase(nil)
	}
	if env == nil {
		return nil, fmt.Errorf("environment cannot be nil")
	}
	if c.Epsilon == nil {
		c.Epsilon = common.DefaultDecaySchedule()
	}
	c.Base.Tracker.TrackValue("epsilon", c.Epsilon.Initial())

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
		Epsilon:         c.Epsilon,
		epsilon:         c.Epsilon.Initial(),
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

	return nil
}

// Action selects the best known action for the given state.
func (a *Agent) Action(state *tensor.Dense) (action int, err error) {
	a.steps++
	a.Tracker.TrackValue("epsilon", a.epsilon)
	if common.RandFloat32(float32(0.0), float32(1.0)) < a.epsilon {
		// explore
		action, err = a.env.SampleAction()
		if err != nil {
			return
		}
		return
	}
	return
}
