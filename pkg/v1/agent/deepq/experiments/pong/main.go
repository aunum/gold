package main

import (
	"github.com/aunum/gold/pkg/v1/agent/deepq"
	"github.com/aunum/gold/pkg/v1/common"
	"github.com/aunum/gold/pkg/v1/common/require"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	"github.com/aunum/log"
)

// NOTE: this is not yet proven to converge.
func main() {
	log.Warning("these given parameters are not yet proven to converge")
	s, err := envv1.NewLocalServer(envv1.GymServerConfig)
	require.NoError(err)
	defer s.Close()

	env, err := s.Make("Pong-v0",
		envv1.WithWrapper(envv1.DefaultAtariWrapper),
		envv1.WithNormalizer(envv1.NewReshapeNormalizer([]int{1, 1, 84, 84})),
	)
	require.NoError(err)

	agentConfig := deepq.DefaultAgentConfig
	agentConfig.PolicyConfig = deepq.DefaultAtariPolicyConfig
	agent, err := deepq.NewAgent(agentConfig, env)
	require.NoError(err)

	agent.View()

	numEpisodes := 20000
	agent.Epsilon = common.DefaultDecaySchedule(common.WithDecayRate(0.9997))
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
