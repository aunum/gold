package main

import (
	"github.com/pbarker/go-rl/pkg/v1/agent/deepq"
	"github.com/pbarker/go-rl/pkg/v1/agent/nes"
	"github.com/pbarker/go-rl/pkg/v1/common/require"
	"github.com/pbarker/go-rl/pkg/v1/dense"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/pbarker/go-rl/pkg/v1/track"
	"github.com/pbarker/log"
	"gorgonia.org/tensor"
)

func main() {
	server, err := envv1.NewLocalServer(envv1.GymServerConfig)
	require.NoError(err)
	defer server.Resource.Close()

	env, err := server.Make("CartPole-v0", envv1.WithNormalizer(envv1.NewExpandDimsNormalizer(0)))
	require.NoError(err)

	blackBox := NewSphereBlackBox()
	config := &nes.EvolverConfig{
		EvolverHyperparameters: nes.DefaultEvolverHyperparameters,
		BlackBox:               blackBox,
	}
	evolver := nes.NewEvolver(config)
	require.NoError(err)

}

// SphereBlackBox is a sphere environment runner.
type SphereBlackBox struct {
	numEpisodes int
	server      *envv1.Server
	envName     string
}

// NewSphereBlackBox returns a new sphere black box.
func NewSphereBlackBox() *SphereBlackBox {
	return &SphereBlackBox{}
}

// Run env.
func (s *SphereBlackBox) Run(weights *tensor.Dense) (reward float32, err error) {
	env, err := s.server.Make(s.envName, envv1.WithNormalizer(envv1.NewExpandDimsNormalizer(0)))
	if err != nil {
		return reward, err
	}
	defer env.Close()

	agent, err := nes.NewAgent(nes.DefaultAgentConfig, env)
	if err != nil {
		return reward, err
	}
	for _, episode := range agent.MakeEpisodes(s.numEpisodes) {
		init, err := env.Reset()
		require.NoError(err)

		state := init.Observation

		score := episode.TrackScalar("score", 0, track.WithAggregator(track.Max))

		for _, timestep := range episode.Steps(env.MaxSteps()) {
			action, err := agent.Action(state)
			require.NoError(err)

			outcome, err := env.Step(action)
			require.NoError(err)

			if outcome.Done {
				outcome.Reward = -outcome.Reward
			}
			score.Inc(outcome.Reward)

			event := deepq.NewEvent(state, action, outcome)
			agent.Remember(event)

			err = agent.Learn()
			require.NoError(err)

			timestep.Log()

			if outcome.Done {
				log.Successf("Episode %d finished after %d timesteps", episode.I, timestep.I+1)
				break
			}
			state = outcome.Observation
		}
	}
	return
}

// InitWeights for the test.
func (s *SphereBlackBox) InitWeights() *tensor.Dense {
	return dense.RandN(tensor.Float32, 3)
}
