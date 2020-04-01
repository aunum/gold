package main

import (
	"github.com/aunum/gold/pkg/v1/agent/her"
	"github.com/aunum/gold/pkg/v1/common"
	"github.com/aunum/gold/pkg/v1/common/require"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	modelv1 "github.com/aunum/goro/pkg/v1/model"
	"github.com/aunum/gold/pkg/v1/track"
	"github.com/aunum/log"

	g "gorgonia.org/gorgonia"
)

type test struct {
	env         string
	learnRate   float64
	numEpisodes int
	decayRate   float32
	batchSize   int
	loss        modelv1.Loss
}

var (
	test10 = &test{
		env:         "BitFlipper10-v0",
		learnRate:   0.0005,
		numEpisodes: 10000,
		decayRate:   0.99995,
		batchSize:   128,
		loss:        modelv1.MSE,
	}
	test15 = &test{
		env:         "BitFlipper15-v0",
		learnRate:   0.0001,
		numEpisodes: 10000,
		decayRate:   0.99997,
		batchSize:   128,
		loss:        modelv1.MSE,
	}
	test20 = &test{
		env:         "BitFlipper20-v0",
		learnRate:   0.00008,
		numEpisodes: 20000,
		decayRate:   0.99998,
		batchSize:   128,
		loss:        modelv1.MSE,
	}
)

func main() {
	runTest(test15)
}

func runTest(t *test) {
	s, err := envv1.NewLocalServer(envv1.GymServerConfig)
	require.NoError(err)
	defer s.Close()

	env, err := s.Make(t.env,
		envv1.WithNormalizer(envv1.NewExpandDimsNormalizer(0)),
		envv1.WithGoalNormalizer(envv1.NewExpandDimsNormalizer(0)),
	)
	require.NoError(err)

	agentConfig := her.DefaultAgentConfig
	agentConfig.PolicyConfig.BatchSize = t.batchSize
	agentConfig.PolicyConfig.Optimizer = g.NewAdamSolver(
		g.WithLearnRate(t.learnRate),
		g.WithBatchSize(float64(t.batchSize)),
	)
	agentConfig.PolicyConfig.Loss = t.loss

	agent, err := her.NewAgent(agentConfig, env)
	require.NoError(err)

	agent.View()

	numEpisodes := t.numEpisodes
	agent.Epsilon = common.DefaultDecaySchedule(common.WithDecayRate(t.decayRate))
	for _, episode := range agent.MakeEpisodes(numEpisodes) {
		init, err := env.Reset()
		require.NoError(err)

		state := init.Observation

		success := episode.TrackScalar("success", 0, track.WithAggregator(track.DefaultRateAggregator))

		episodeEvents := her.Events{}
		for _, timestep := range episode.Steps(env.MaxSteps()) {
			log.Infov("state", state.Data())
			log.Infov("goal", init.Goal.Data())
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
				log.Infov("expected goal", init.Goal.Data())
				log.Successf("Episode %d finished after %d timesteps, with success of %v", episode.I, timestep.I+1, success.Scalar())
				break
			}
			state = outcome.Observation
		}
		agent.Remember(episodeEvents...)

		// apply hindsight if episode failed.
		if success.Scalar() == 0 {
			err = agent.Hindsight(episodeEvents)
			require.NoError(err)
		}
		episode.Log()
	}
	agent.Wait()
	env.Close()
}
