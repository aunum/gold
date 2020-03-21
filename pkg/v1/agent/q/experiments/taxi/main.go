package main

import (
	"github.com/aunum/gold/pkg/v1/agent/q"
	"github.com/aunum/gold/pkg/v1/common/require"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	"github.com/aunum/log"
)

func main() {
	s, err := envv1.FindOrCreate(envv1.GymServerConfig)
	require.NoError(err)
	defer s.Close()

	env, err := s.Make("Taxi-v3")
	require.NoError(err)

	agent := q.NewAgent(q.DefaultAgentConfig, env)

	agent.View()

	numEpisodes := 5000
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

			err = agent.Learn(action, state, outcome)
			require.NoError(err)

			if outcome.Done {
				log.Successf("Episode %d finished after %d timesteps with a score of %v", episode.I, timestep.I+1, score.Scalar())
				break
			}
			state = outcome.Observation
		}
		episode.Log()
	}
	agent.Wait()
	env.End()
}
