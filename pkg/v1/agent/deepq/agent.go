package deepq

import (
	"fmt"

	"github.com/pbarker/go-rl/pkg/v1/dense"
	"github.com/pbarker/go-rl/pkg/v1/model"

	agentv1 "github.com/pbarker/go-rl/pkg/v1/agent"
	"github.com/pbarker/go-rl/pkg/v1/common"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/pbarker/logger"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Agent is a dqn agent.
type Agent struct {
	// Base for the agent.
	*agentv1.Base

	// Hyperparameters for the dqn agent.
	*Hyperparameters

	Policy            model.Model
	TargetPolicy      model.Model
	Epsilon           common.Schedule
	env               *envv1.Env
	epsilon           float32
	updateTargetSteps int

	memory *Memory
	steps  int
}

// Hyperparameters for the dqn agent.
type Hyperparameters struct {
	// Gamma is the discount factor (0≤γ≤1). It determines how much importance we want to give to future
	// rewards. A high value for the discount factor (close to 1) captures the long-term effective award, whereas,
	// a discount factor of 0 makes our agent consider only immediate reward, hence making it greedy.
	Gamma float32

	// Epsilon is the rate at which the agent should exploit vs explore.
	Epsilon common.Schedule

	// ReplayBatchSize determines how large a batch is replayed from memory.
	ReplayBatchSize int

	// UpdateTargetSteps determins how often the target network updates its parameters.
	UpdateTargetSteps int
}

// DefaultHyperparameters are the default hyperparameters.
var DefaultHyperparameters = &Hyperparameters{
	Epsilon:           common.DefaultDecaySchedule(),
	Gamma:             0.95,
	ReplayBatchSize:   20,
	UpdateTargetSteps: 100,
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
		c.Base = agentv1.NewBase(nil)
	}
	if env == nil {
		return nil, fmt.Errorf("environment cannot be nil")
	}
	if c.Epsilon == nil {
		c.Epsilon = common.DefaultDecaySchedule()
	}
	policy, err := MakePolicy("online", c.PolicyConfig, c.Base, env)
	if err != nil {
		return nil, err
	}
	targetPolicy, err := MakePolicy("target", c.PolicyConfig, c.Base, env)
	if err != nil {
		return nil, err
	}
	c.Base.Tracker.TrackValue("epsilon", c.Epsilon.Initial())
	return &Agent{
		Base:              c.Base,
		Hyperparameters:   c.Hyperparameters,
		memory:            NewMemory(),
		Policy:            policy,
		TargetPolicy:      targetPolicy,
		Epsilon:           c.Epsilon,
		epsilon:           c.Epsilon.Initial(),
		env:               env,
		updateTargetSteps: c.UpdateTargetSteps,
	}, nil
}

// Learn the agent.
func (a *Agent) Learn() error {
	// logger.Infof("batch size: %v", a.memory.Len())
	if a.memory.Len() < a.ReplayBatchSize {
		return nil
	}
	logger.Info("learning")
	batch, err := a.memory.Sample(a.ReplayBatchSize)
	if err != nil {
		return err
	}
	for i, event := range batch {
		fmt.Printf("\n---------- %v \n", i)
		event.Print()
		logger.Infov("i", event.i)
		qUpdate := float32(event.Reward)
		if !event.Done {
			prediction, err := a.TargetPolicy.Predict(event.Observation)
			if err != nil {
				return err
			}
			qValues := prediction.(*tensor.Dense)
			logger.Infov("next qvalues", qValues)
			nextMax, err := dense.AMaxF32(qValues, 1)
			if err != nil {
				return err
			}
			logger.Infov("next max", nextMax)
			qUpdate = event.Reward + a.Gamma*nextMax
		}
		logger.Infof("qUpdate %v for action %v", qUpdate, event.Action)
		prediction, err := a.Policy.Predict(event.State)
		if err != nil {
			return err
		}
		qValues := prediction.(*tensor.Dense)
		logger.Infov("current qvalues", qValues)
		qValues.Set(event.Action, qUpdate)
		logger.Infov("x", event.State)
		logger.Infov("y", qValues)
		err = a.Policy.Fit(event.State, qValues)
		if err != nil {
			return err
		}
		// for i := 0; i < 10; i++ {
		// 	err = a.Policy.Fit(event.State, qValues)
		// 	if err != nil {
		// 		return err
		// 	}
		// }
		fpred, err := a.Policy.Predict(event.State)
		if err != nil {
			return err
		}
		fqValues := fpred.(*tensor.Dense)
		logger.Infov("final qvalues", fqValues)
		fmt.Printf("---------- %v \n\n", i)
	}
	a.epsilon = a.Epsilon.Value()

	err = a.updateTarget()
	if err != nil {
		return err
	}
	return nil
}

func (a *Agent) updateTarget() error {
	if a.steps%a.updateTargetSteps == 0 {
		logger.Info("###################################")
		logger.Infof("updating target model - steps %v target %v", a.steps, a.updateTargetSteps)
		targetLearnables := a.TargetPolicy.Learnables()
		for i, layer := range a.Policy.Learnables() {
			err := g.Let(targetLearnables[i], layer.Clone().(*g.Node).Value())
			if err != nil {
				return err
			}
		}
		logger.Info("online learnables: ", a.Policy.Learnables())
		for _, layer := range a.Policy.Learnables() {
			logger.Infov(layer.Name(), layer.Value())
		}
		logger.Info("-------")
		logger.Info("target learnables: ", a.Policy.Learnables())
		for _, layer := range a.TargetPolicy.Learnables() {
			logger.Infov(layer.Name(), layer.Value())
		}
		logger.Info("###################################")
	}
	return nil
}

// Action selects the best known action for the given state.
func (a *Agent) Action(state *tensor.Dense) (action int, err error) {
	a.steps++
	logger.Infov("epsilon", a.epsilon)
	a.Tracker.TrackValue("epsilon", a.epsilon)
	if common.RandFloat32(float32(0.0), float32(1.0)) < a.epsilon {
		// explore
		action, err = a.env.SampleAction()
		if err != nil {
			return
		}
		return
	}
	action, err = a.action(state)
	return
}

func (a *Agent) action(state *tensor.Dense) (action int, err error) {
	fmt.Println("************")
	prediction, err := a.Policy.Predict(state)
	if err != nil {
		return
	}
	qValues := prediction.(*tensor.Dense)
	logger.Infov("qvalues", qValues)
	actionIndex, err := qValues.Argmax(1)
	if err != nil {
		return action, err
	}
	logger.Infov("actionIndex", actionIndex)
	action = actionIndex.GetI(0)
	logger.Infov("best action", action)
	fmt.Println("************")
	return
}

// Remember an event.
func (a *Agent) Remember(event *Event) {
	a.memory.PushFront(event)
}
