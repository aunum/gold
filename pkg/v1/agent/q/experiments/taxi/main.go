package main

import (
	"github.com/pbarker/go-rl/pkg/v1/agent/q"
	"github.com/pbarker/go-rl/pkg/v1/common/require"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/pbarker/go-rl/pkg/v1/track"
	"github.com/pbarker/log"
)

func main() {
	s, err := envv1.NewLocalServer(envv1.GymServerConfig)
	require.NoError(err)
	defer s.Resource.Close()

	env, err := s.Make("Taxi-v3")
	require.NoError(err)

	agent := q.NewAgent(q.DefaultAgentConfig, env)

	agent.View()

	numEpisodes := 5000
	for _, episode := range agent.MakeEpisodes(numEpisodes) {
		init, err := env.Reset()
		require.NoError(err)

		state := init.Observation

		score := episode.TrackScalar("score", 0, track.WithAggregator(track.Max))

		for _, timestep := range episode.Steps(env.MaxSteps()) {
			log.Infovb("state", state)
			action, err := agent.Action(state)
			require.NoError(err)

			outcome, err := env.Step(action)
			require.NoError(err)

			score.Inc(outcome.Reward)

			err = agent.Learn(action, state, outcome)
			require.NoError(err)

			if outcome.Done {
				log.Successf("Episode %d finished after %d timesteps with a score of %v", episode.I, timestep.I+1, score)
				break
			}
			state = outcome.Observation
		}
		episode.Log()
	}
	agent.Wait()
	env.End()
}
