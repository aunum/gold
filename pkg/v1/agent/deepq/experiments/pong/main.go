package main

import (
	"github.com/aunum/gold/pkg/v1/agent/deepq"
	"github.com/aunum/gold/pkg/v1/common"
	"github.com/aunum/gold/pkg/v1/common/require"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	modelv1 "github.com/aunum/gold/pkg/v1/model"
	"github.com/aunum/gold/pkg/v1/model/layer"
	"github.com/aunum/log"

	g "gorgonia.org/gorgonia"
)

func main() {
	s, err := envv1.NewLocalServer(envv1.GymServerConfig)
	require.NoError(err)
	defer s.Close()

	env, err := s.Make("Pong-v0",
		envv1.WithWrapper(envv1.DefaultAtariWrapper),
		envv1.WithNormalizer(envv1.NewReshapeNormalizer([]int{1, 84 * 84})),
	)
	require.NoError(err)

	// DefaultFCLayerBuilder is a default fully connected layer builder.
	layerBuilder := func(x, y *modelv1.Input) []layer.Config {
		return []layer.Config{
			layer.FC{Input: x.Squeeze()[0], Output: 256},
			layer.FC{Input: 256, Output: 24},
			layer.FC{Input: 24, Output: y.Squeeze()[0], Activation: layer.Linear},
		}
	}
	agentConfig := deepq.DefaultAgentConfig
	policyConfig := &deepq.PolicyConfig{
		Loss:         modelv1.MSE,
		Optimizer:    g.NewAdamSolver(g.WithLearnRate(0.0005)),
		LayerBuilder: layerBuilder,
		BatchSize:    20,
		Track:        true,
	}
	agentConfig.PolicyConfig = policyConfig
	agent, err := deepq.NewAgent(deepq.DefaultAgentConfig, env)
	require.NoError(err)

	agent.View()

	numEpisodes := 200
	agent.Epsilon = common.DefaultDecaySchedule(common.WithDecayRate(0.9995))
	for _, episode := range agent.MakeEpisodes(numEpisodes) {
		init, err := env.Reset()
		require.NoError(err)

		state := init.Observation

		score := episode.TrackScalar("score", 0)

		for _, timestep := range episode.Steps(env.MaxSteps()) {
			action, err := agent.Action(state)
			require.NoError(err)

			outcome, err := env.Step(action)
			require.NoError(err)

			score.Inc(outcome.Reward)

			event := deepq.NewEvent(state, action, outcome)
			agent.Remember(event)

			err = agent.Learn()
			require.NoError(err)

			if outcome.Done {
				log.Successf("Episode %d finished after %d timesteps with a score of %v", episode.I, timestep.I+1, score.Scalar())
				break
			}
			state = outcome.Observation

			err = agent.Render(env)
			require.NoError(err)
		}
		episode.Log()
	}
	agent.Wait()
	env.End()
}
