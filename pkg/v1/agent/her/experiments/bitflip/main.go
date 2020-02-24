package main

import (
	"github.com/pbarker/go-rl/pkg/v1/agent/her"
	"github.com/pbarker/go-rl/pkg/v1/common"
	"github.com/pbarker/go-rl/pkg/v1/common/require"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	modelv1 "github.com/pbarker/go-rl/pkg/v1/model"
	"github.com/pbarker/go-rl/pkg/v1/track"
	"github.com/pbarker/log"

	g "gorgonia.org/gorgonia"
)

// Test are the params for a test.
type Test struct {
	env         string
	learnRate   float64
	numEpisodes int
	decayRate   float32
	batchSize   int
	loss        modelv1.Loss
}

var (
	test10 = &Test{
		env:         "BitFlipper10-v0",
		learnRate:   0.0005,
		numEpisodes: 10000,
		decayRate:   0.99995,
		batchSize:   128,
		loss:        modelv1.MSE,
	}
	test15 = &Test{
		env:         "BitFlipper15-v0",
		learnRate:   0.0001,
		numEpisodes: 10000,
		decayRate:   0.99997,
		batchSize:   128,
		loss:        modelv1.MSE,
	}
	test20 = &Test{
		env:         "BitFlipper20-v0",
		learnRate:   0.00008,
		numEpisodes: 20000,
		decayRate:   0.99998,
		batchSize:   128,
		loss:        modelv1.MSE,
	}
)

func main() {
	test(test20)
}

func test(test *Test) {
	s, err := envv1.NewLocalServer(envv1.GymServerConfig)
	require.NoError(err)
	defer s.Resource.Close()

	env, err := s.Make(test.env, envv1.WithNormalizer(envv1.NewExpandDimsNormalizer(0)))
	require.NoError(err)

	agentConfig := her.DefaultAgentConfig
	agentConfig.PolicyConfig.BatchSize = test.batchSize
	agentConfig.PolicyConfig.Optimizer = g.NewAdamSolver(
		g.WithLearnRate(test.learnRate),
		g.WithBatchSize(float64(test.batchSize)),
	)
	agentConfig.PolicyConfig.Loss = test.loss
	agent, err := her.NewAgent(agentConfig, env)
	require.NoError(err)

	agent.View()

	numEpisodes := test.numEpisodes
	agent.Epsilon = common.DefaultDecaySchedule(common.WithDecayRate(test.decayRate))
	for _, episode := range agent.MakeEpisodes(numEpisodes) {
		init, err := env.Reset()
		require.NoError(err)
		state := init.Observation
		log.Infov("state", state.Data())
		log.Infov("goal", init.Goal.Data())

		success := episode.TrackScalar("success", 0, track.WithAggregator(track.DefaultRateAggregator))
		episodeEvents := her.Events{}
		for _, timestep := range episode.Steps(env.MaxSteps()) {
			log.BreakPound()
			log.Infov("action state", state.Data())
			log.Infov("action goal", init.Goal.Data())
			action, err := agent.Action(state, init.Goal)
			require.NoError(err)

			outcome, err := env.Step(action)
			require.NoError(err)

			event := her.NewEvent(state, init.Goal, outcome)
			episodeEvents = append(episodeEvents, event)
			if outcome.Done {
				if outcome.Reward == 0 {
					success.Set(1)
				}
				log.Infov("final state", outcome.Observation.Data())
				log.Successf("Episode %d finished after %d timesteps, with success of %v", episode.I, timestep.I+1, success.Scalar())
				break
			}
			log.Infov("setting state", outcome.Observation.Data())
			state = outcome.Observation
		}
		agent.Remember(episodeEvents...)
		if success.Scalar() == 0 {
			err = agent.Hindsight(episodeEvents)
			require.NoError(err)
		}
		episode.Log()
	}
	agent.Wait()
	env.Close()
}
