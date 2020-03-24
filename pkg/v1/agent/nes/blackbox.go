package nes

import (
	"sync"

	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	"github.com/aunum/gold/pkg/v1/common/num"
	"github.com/aunum/gold/pkg/v1/common/require"
	"github.com/aunum/gold/pkg/v1/dense"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	"github.com/aunum/log"
	"gorgonia.org/tensor"
	t "gorgonia.org/tensor"
)

// BlackBox function we wish to optimize.
type BlackBox interface {
	// Run the black box.
	Run(weights *t.Dense) (reward float32, err error)

	// RunAsync the black box.
	RunAsync(populationID int, weights *tensor.Dense, results chan BlackBoxResult, wg *sync.WaitGroup)

	// Initialize the weights.
	InitWeights() *t.Dense
}

// BlackBoxResult is the result of a black box run.
type BlackBoxResult struct {
	// Reward from black box run.
	Reward float32

	// Error from run.
	Err error

	// PopulationID is the ID of the population.
	PopulationID int

	// Solved tells if the agent solved the problem.
	Solved bool

	// Weights if solved.
	Weights *tensor.Dense
}

// SphereBlackBoxConfig is the sphere black box config.
type SphereBlackBoxConfig struct {
	// NumEpisodes is the number of episodes.
	NumEpisodes int

	// EnvName is the environment name.
	EnvName string

	// AgentConfig is the agent config.
	AgentConfig *AgentConfig

	// Logger for the box.
	Logger *log.Logger

	// SolvedChecker checks if the environment is solved.
	SolvedChecker SolvedChecker
}

// SolvedChecker checks if the environment is solved.
type SolvedChecker func(reward float32) bool

// DefaultSphereBlackBoxConfig is the default config for a sphere black box.
var DefaultSphereBlackBoxConfig = &SphereBlackBoxConfig{
	NumEpisodes: 100,
	EnvName:     "CartPole-v0",
	AgentConfig: DefaultAgentConfig,
	Logger:      log.DefaultLogger,
	SolvedChecker: func(reward float32) bool {
		if reward >= 195 {
			return true
		}
		return false
	},
}

// SphereBlackBox is a sphere environment runner.
type SphereBlackBox struct {
	numEpisodes   int
	server        *envv1.Server
	envName       string
	agentConfig   *AgentConfig
	weightShape   tensor.Shape
	logger        *log.Logger
	solvedChecker SolvedChecker
}

// NewSphereBlackBox returns a new sphere black box.
func NewSphereBlackBox(config *SphereBlackBoxConfig, server *envv1.Server) (*SphereBlackBox, error) {
	env, err := server.Make(config.EnvName)
	if err != nil {
		return nil, err
	}
	defer env.Close()
	weightShape := weightShape(env)
	if config.Logger == nil {
		config.Logger = log.DefaultLogger
	}
	return &SphereBlackBox{
		numEpisodes:   config.NumEpisodes,
		server:        server,
		envName:       config.EnvName,
		agentConfig:   config.AgentConfig,
		weightShape:   weightShape,
		logger:        config.Logger,
		solvedChecker: config.SolvedChecker,
	}, nil
}

// Run env.
func (s *SphereBlackBox) Run(weights *tensor.Dense) (reward float32, err error) {
	env, err := s.server.Make(s.envName, envv1.WithNormalizer(envv1.NewExpandDimsNormalizer(0)))
	if err != nil {
		return reward, err
	}
	defer func() {
		err = env.Close()
		if err != nil {
			s.logger.Error(err)
		}
	}()

	base := agentv1.NewBase("nes worker", agentv1.WithLogger(s.logger), agentv1.WithoutTracker(), agentv1.WithoutServer())
	agent, err := NewAgent(s.agentConfig, env, base)
	if err != nil {
		return reward, err
	}
	err = agent.SetWeights(weights)
	if err != nil {
		return
	}

	scores := []float32{}
	for episode := 0; episode <= s.numEpisodes; episode++ {
		init, err := env.Reset()
		require.NoError(err)

		state := init.Observation

		var score float32
		for step := 0; step <= env.MaxSteps(); step++ {
			action, err := agent.Action(state)
			require.NoError(err)

			outcome, err := env.Step(action)
			require.NoError(err)

			if outcome.Done {
				outcome.Reward = -outcome.Reward
			}
			score += outcome.Reward

			if outcome.Done {
				break
			}
			state = outcome.Observation
		}
		scores = append(scores, score)
	}
	reward = num.Mean(scores)
	return
}

// RunAsync runs the black box async
func (s *SphereBlackBox) RunAsync(populationID int, weights *tensor.Dense, results chan BlackBoxResult, wg *sync.WaitGroup) {
	defer wg.Done()
	reward, err := s.Run(weights)
	res := BlackBoxResult{Reward: reward, Err: err, PopulationID: populationID}
	if s.solvedChecker(reward) {
		res.Solved = true
		res.Weights = weights
	}
	results <- res
}

// InitWeights for the test.
func (s *SphereBlackBox) InitWeights() *tensor.Dense {
	return dense.RandN(tensor.Float32, s.weightShape...)
}

// TODO: this is somewhat fragile.
func weightShape(e *envv1.Env) tensor.Shape {
	obs := e.ObservationSpaceShape()
	log.Debugv("obv space shape", obs)
	as := envv1.PotentialsShape(e.ActionSpace)
	log.Debugv("action space shape", as)
	return []int{
		dense.SqueezeShape(e.ObservationSpaceShape())[0],
		dense.SqueezeShape(envv1.PotentialsShape(e.ActionSpace))[0],
	}
}
