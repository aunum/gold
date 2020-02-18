package main

import (
	"fmt"

	"github.com/pbarker/go-rl/pkg/v1/agent/her"
	"github.com/pbarker/go-rl/pkg/v1/common"
	"github.com/pbarker/go-rl/pkg/v1/common/require"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/pbarker/log"
)

func main() {
	s, err := envv1.NewLocalServer(envv1.GymServerConfig)
	require.NoError(err)
	defer s.Resource.Close()

	env, err := s.Make("BitFlipper-v0")
	require.NoError(err)

	fmt.Println(env)

	agent, err := her.NewAgent(her.DefaultAgentConfig, env)
	require.NoError(err)

	agent.View()

	numEpisodes := 200
	agent.Epsilon = common.DefaultDecaySchedule(common.WithDecayRate(0.9995))
	for _, episode := range agent.MakeEpisodes(numEpisodes) {
		init, err := env.Reset()
		require.NoError(err)
		state := init.Observation
		log.Infov("state", state.Data())
		log.Infov("goal", init.Observation.Data())

		success := episode.TrackScalar("success", 0)
		episodeEvents := her.Events{}
		for _, timestep := range episode.Steps(env.MaxSteps()) {
			action, err := agent.Action(state, init.Goal)
			require.NoError(err)

			outcome, err := env.Step(action)
			require.NoError(err)

			event := her.NewEvent(state, init.Goal, outcome)
			episodeEvents = append(episodeEvents, event)

			timestep.Log()

			if outcome.Done {
				if outcome.Reward == 0 {
					success.Set(1)
				}
				log.Successf("Episode %d finished after %d timesteps, with success of %v", episode.I, timestep.I+1, success.Scalar())
				break
			}
			state = outcome.Observation
		}
		agent.Remember(episodeEvents...)
		if success.Scalar() == 0 {
			agent.Hindsight(episodeEvents)
		}
		err = agent.Learn()
		require.NoError(err)
		episode.Log()
	}
	agent.Wait()
	env.End()
}
