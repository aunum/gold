// Package her is an agent implementation of the Hindsight Experience Replay algorithm.
package her

import (
	"fmt"
	"math/rand"

	"github.com/aunum/gold/pkg/v1/dense"
	"github.com/aunum/goro/pkg/v1/model"
	"github.com/aunum/gold/pkg/v1/track"

	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	"github.com/aunum/gold/pkg/v1/common"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	"github.com/aunum/log"
	"gorgonia.org/tensor"
)

// Agent is a dqn+her agent.
type Agent struct {
	// Base for the agent.
	*agentv1.Base

	// Hyperparameters for the dqn+her agent.
	*Hyperparameters

	Policy            model.Model
	TargetPolicy      model.Model
	Epsilon           common.Schedule
	env               *envv1.Env
	epsilon           *track.TrackedScalarValue
	updateTargetSteps int
	batchSize         int

	memory           *Memory
	steps            int
	episodes         int
	successfulReward float32
}

// Hyperparameters for the dqn+her agent.
type Hyperparameters struct {
	// Gamma is the discount factor (0≤γ≤1). It determines how much importance we want to give to future
	// rewards. A high value for the discount factor (close to 1) captures the long-term effective award, whereas,
	// a discount factor of 0 makes our agent consider only immediate reward, hence making it greedy.
	Gamma float32

	// Epsilon is the rate at which the agent should exploit vs explore.
	Epsilon common.Schedule

	// UpdateTargetEpisodes determines how often the target network updates its parameters.
	UpdateTargetEpisodes int
}

// DefaultHyperparameters are the default hyperparameters.
var DefaultHyperparameters = &Hyperparameters{
	Epsilon:              common.DefaultDecaySchedule(),
	Gamma:                0.9,
	UpdateTargetEpisodes: 50,
}

// AgentConfig is the config for a dqn+her agent.
type AgentConfig struct {
	// Base for the agent.
	Base *agentv1.Base

	// Hyperparameters for the agent.
	*Hyperparameters

	// PolicyConfig for the agent.
	PolicyConfig *PolicyConfig

	// SuccessfulReward is the reward for reaching the goal.
	SuccessfulReward float32

	// MemorySize is the size of the memory.
	MemorySize int
}

// DefaultAgentConfig is the default config for a dqn+her agent.
var DefaultAgentConfig = &AgentConfig{
	Hyperparameters:  DefaultHyperparameters,
	PolicyConfig:     DefaultPolicyConfig,
	Base:             agentv1.NewBase("HER"),
	SuccessfulReward: 0,
	MemorySize:       1e4,
}

// NewAgent returns a new dqn+her agent.
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
	if c.Epsilon == nil {
		c.Epsilon = common.DefaultDecaySchedule()
	}
	policy, err := MakePolicy("online", c.PolicyConfig, c.Base, env)
	if err != nil {
		return nil, err
	}
	c.PolicyConfig.Track = false
	targetPolicy, err := MakePolicy("target", c.PolicyConfig, c.Base, env)
	if err != nil {
		return nil, err
	}
	epsilon := c.Base.Tracker.TrackValue("epsilon", c.Epsilon.Initial())
	return &Agent{
		Base:              c.Base,
		Hyperparameters:   c.Hyperparameters,
		memory:            NewMemory(c.MemorySize),
		Policy:            policy,
		TargetPolicy:      targetPolicy,
		Epsilon:           c.Epsilon,
		epsilon:           epsilon.(*track.TrackedScalarValue),
		env:               env,
		updateTargetSteps: c.UpdateTargetEpisodes,
		batchSize:         c.PolicyConfig.BatchSize,
		successfulReward:  c.SuccessfulReward,
	}, nil
}

// Learn the agent.
func (a *Agent) Learn() error {
	if a.memory.Len() < a.batchSize {
		return nil
	}
	batch, err := a.memory.Sample(a.batchSize)
	if err != nil {
		return err
	}
	batchStates := []*tensor.Dense{}
	batchQValues := []*tensor.Dense{}
	for _, event := range batch {
		qUpdate := float32(event.Reward)
		if !event.Done {
			obvGoal, err := event.Observation.Concat(1, event.Goal)
			if err != nil {
				return err
			}
			targetPred, err := a.TargetPolicy.Predict(obvGoal)
			if err != nil {
				return err
			}
			qValues := targetPred.(*tensor.Dense)
			nextMax, err := dense.AMaxF32(qValues, 1)
			if err != nil {
				return err
			}
			qUpdate = event.Reward + (a.Gamma * nextMax)
		}

		stateGoal, err := event.State.Concat(1, event.Goal)
		if err != nil {
			return err
		}
		prediction, err := a.Policy.Predict(stateGoal)
		if err != nil {
			return err
		}
		qValues := prediction.(*tensor.Dense)
		qValues.Set(event.Action, qUpdate)
		batchStates = append(batchStates, stateGoal)
		batchQValues = append(batchQValues, qValues)
	}
	states, err := dense.Concat(0, batchStates...)
	if err != nil {
		return err
	}
	qValues, err := dense.Concat(0, batchQValues...)
	if err != nil {
		return err
	}
	err = a.Policy.FitBatch(states, qValues)
	if err != nil {
		return err
	}
	a.episodes++
	err = a.updateTarget()
	if err != nil {
		return err
	}
	return nil
}

// updateTarget copies the weights from the online network to the target network on the provided interval.
func (a *Agent) updateTarget() error {
	if a.episodes%a.UpdateTargetEpisodes == 0 {
		log.Infof("updating target model - current steps %v target update %v", a.steps, a.updateTargetSteps)
		err := a.Policy.(*model.Sequential).CloneLearnablesTo(a.TargetPolicy.(*model.Sequential))
		if err != nil {
			return err
		}
	}
	return nil
}

// Action selects the best known action for the given state.
func (a *Agent) Action(state, goal *tensor.Dense) (action int, err error) {
	a.steps++
	defer func() { a.epsilon.Set(a.Epsilon.Value()) }()
	if rand.Float64() < a.epsilon.Scalar() {
		// explore
		action, err = a.env.SampleAction()
		if err != nil {
			return
		}
		log.Infov("taking random action", action)
		return
	}
	action, err = a.action(state, goal)
	log.Infov("taking action", action)
	return
}

func (a *Agent) action(state, goal *tensor.Dense) (action int, err error) {
	stateGoal, err := state.Concat(1, goal)
	if err != nil {
		return action, err
	}
	prediction, err := a.Policy.Predict(stateGoal)
	if err != nil {
		return
	}
	qValues := prediction.(*tensor.Dense)
	actionIndex, err := qValues.Argmax(1)
	if err != nil {
		return action, err
	}
	action = actionIndex.GetI(0)
	return
}

// Remember events.
func (a *Agent) Remember(event ...*Event) {
	a.memory.Remember(event...)
}

// Hindsight applies hindsight to the memory.
func (a *Agent) Hindsight(episodeEvents Events) error {
	log.Debug("running hindsight")
	altBatch := episodeEvents.Copy()
	finalEvent := altBatch[len(altBatch)-1]
	for i, event := range altBatch {
		event.Goal = finalEvent.Outcome.Observation
		if i == len(altBatch)-1 {
			event.Reward = a.successfulReward
			event.Done = true
		}
		a.memory.Remember(event)
		err := a.Learn()
		if err != nil {
			return err
		}
	}
	return nil
}
