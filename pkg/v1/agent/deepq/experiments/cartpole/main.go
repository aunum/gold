package main

import (
	"github.com/pbarker/go-rl/pkg/v1/agent/deepq"
	"github.com/pbarker/go-rl/pkg/v1/common/require"
	sphere "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/pbarker/logger"
)

func main() {
	s, err := sphere.NewLocalServer(sphere.GymServerConfig)
	require.NoError(err)
	defer s.Resource.Close()

	env, err := s.Make("CartPole-v0")
	require.NoError(err)

	agent, err := deepq.NewAgent(deepq.DefaultAgentConfig, env)
	require.NoError(err)

	numEpisodes := 200
	logger.Infof("running for %d episodes", numEpisodes)
	for i := 0; i <= numEpisodes; i++ {
		state, err := env.Reset()
		require.NoError(err)

		for ts := 0; ts <= int(env.MaxEpisodeSteps); ts++ {
			action, err := agent.Action(state)
			require.NoError(err)

			outcome, err := env.Step(action)
			require.NoError(err)

			event := deepq.NewEvent(state, action, outcome)
			agent.Remember(event)

			err = agent.Learn()
			require.NoError(err)

			if outcome.Done {
				logger.Successf("Episode %d finished after %d timesteps", i, ts+1)
				break
			}
			state = outcome.Observation
		}
	}
	env.End()
}
