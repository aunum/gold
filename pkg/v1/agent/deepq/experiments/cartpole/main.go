package main

import (
	"github.com/pbarker/go-rl/pkg/v1/agent/deepq"
	"github.com/pbarker/go-rl/pkg/v1/common"
	"github.com/pbarker/go-rl/pkg/v1/common/require"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/pbarker/go-rl/pkg/v1/track"
	"github.com/pbarker/log"
)

func main() {
	s, err := envv1.NewLocalServer(envv1.GymServerConfig)
	require.NoError(err)
	defer s.Resource.Close()

	env, err := s.Make("CartPole-v1", envv1.WithNormalizer(envv1.NewExpandDimsNormalizer(0)))
	require.NoError(err)

	agent, err := deepq.NewAgent(deepq.DefaultAgentConfig, env)
	require.NoError(err)

	agent.View()

	numEpisodes := 200
	agent.Epsilon = common.DefaultDecaySchedule(common.WithDecayRate(0.9995))
	for _, episode := range agent.MakeEpisodes(numEpisodes) {
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
	agent.Wait()
	env.End()
}
