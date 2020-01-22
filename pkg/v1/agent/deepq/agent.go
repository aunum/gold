package deepq

import (
	"fmt"

	"github.com/pbarker/go-rl/pkg/v1/model"

	"github.com/chewxy/math32"
	agentv1 "github.com/pbarker/go-rl/pkg/v1/agent"
	"github.com/pbarker/go-rl/pkg/v1/common"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/pbarker/go-rl/pkg/v1/track"
	"github.com/pbarker/logger"
	"gorgonia.org/tensor"
)

// Agent is a dqn agent.
type Agent struct {
	// Base for the agent.
	*agentv1.Base

	// Hyperparameters for the dqn agent.
	*Hyperparameters

	// Tracker for the agent.
	*track.Tracker

	Policy model.Model
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
	// TODO: these should be a schedule.
	EpsilonMin float32

	// EpsilonMax is the maximum rate at which an agent can explore.
	EpsilonMax float32

	// EpsilonDecay is the rate at which the agent should exploit over explore.
	EpsilonDecay float32

	// ReplayBatchSize determines how large a batch is replayed from memory.
	ReplayBatchSize int
}

// DefaultHyperparameters are the default hyperparameters.
var DefaultHyperparameters = &Hyperparameters{
	Gamma:           0.95,
	Alpha:           0.001,
	EpsilonMin:      0.01,
	EpsilonMax:      1.0,
	EpsilonDecay:    0.995,
	ReplayBatchSize: 30,
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
	Base:            agentv1.NewBase(),
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
	policy, err := MakePolicy("train", c.PolicyConfig, c.Base, env)
	if err != nil {
		return nil, err
	}
	return &Agent{
		Base:            c.Base,
		Hyperparameters: c.Hyperparameters,
		epsilon:         c.EpsilonMax,
		memory:          NewMemory(),
		Policy:          policy,
		Tracker:         c.Base.Tracker,
		env:             env,
	}, nil
}

// Learn the agent.
func (a *Agent) Learn() error {
	logger.Infof("batch size: %v", a.memory.Len())
	if a.memory.Len() < a.ReplayBatchSize {
		return nil
	}
	logger.Info("learning")
	batch, err := a.memory.Sample(a.ReplayBatchSize)
	if err != nil {
		return err
	}
	for _, event := range batch {
		update := float32(event.Reward)
		if !event.Done {
			prediction, err := a.Policy.Predict(event.Observation)
			if err != nil {
				return err
			}
			qValues := prediction.(*tensor.Dense)
			// logger.Info("qvalues: ", qValues)
			maxIndex, err := qValues.Argmax(0)
			nextMax := qValues.GetF32(maxIndex.GetI(0))
			update = (float32(event.Reward) + a.Gamma*nextMax)
		} else {
			update = -update
		}
		prediction, err := a.Policy.Predict(event.State)
		if err != nil {
			return err
		}
		qValues := prediction.(*tensor.Dense)
		qValues.Set(event.Action, update)
		// logger.Info("y: ", qValues)
		err = a.Policy.Fit(event.State, qValues)
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
	logger.Infof("epsilon: %v", a.epsilon)
	if common.RandFloat32(float32(0.0), float32(1.0)) < a.epsilon {
		// explore
		// logger.Info("exploring")
		action, err = a.env.SampleAction()
		if err != nil {
			return
		}
		return
	}
	// logger.Info("exploiting")
	action, err = a.action(state)
	return
}

func (a *Agent) action(state *tensor.Dense) (action int, err error) {
	prediction, err := a.Policy.Predict(state)
	if err != nil {
		return
	}
	qValues := prediction.(*tensor.Dense)
	logger.Info("qvalues: ", qValues)
	actionIndex, err := qValues.Argmax(0)
	if err != nil {
		return action, err
	}
	action = actionIndex.GetI(0)
	logger.Info("best action: ", action)
	return
}

// Remember an event.
func (a *Agent) Remember(event *Event) {
	a.memory.PushFront(event)
}
