package main

import (
	"github.com/pbarker/go-rl/pkg/v1/agent/her"
	"github.com/pbarker/go-rl/pkg/v1/agent/her/experiments/toy/flipper"
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

	env0, err := s.Make("BitFlipper-v0", envv1.WithNormalizer(envv1.NewExpandDimsNormalizer(0)))
	require.NoError(err)

	n := 4
	env := flipper.NewEnv(n, flipper.WithStatic())
	env0.ObservationSpace.GetMultiBinary().N = int32(n)
	env0.ActionSpace.GetDiscrete().N = int32(n)

	agent, err := her.NewAgent(her.DefaultAgentConfig, env0)
	require.NoError(err)

	agent.View()

	numEpisodes := 4000
	agent.Epsilon = common.DefaultDecaySchedule(common.WithDecayRate(0.999))
	for _, episode := range agent.MakeEpisodes(numEpisodes) {
		state, goal := env.Reset()
		log.Infov("state", state.Data())
		log.Infov("goal", goal.Data())

		success := episode.TrackScalar("success", 0, track.WithAggregator(track.DefaultRateAggregator))
		episodeEvents := her.Events{}
		for _, timestep := range episode.Steps(env.MaxSteps()) {
			log.BreakPound()
			log.Infov("action state", state.Data())
			log.Infov("action goal", goal.Data())
			action, err := agent.Action(state, goal)
			require.NoError(err)

			observation, done, reward := env.Step(action)

			outcome := &envv1.Outcome{
				Observation: observation,
				Action:      action,
				Reward:      float32(reward),
				Done:        done,
			}
			event := her.NewEvent(state, goal, outcome)
			episodeEvents = append(episodeEvents, event)
			// agent.Learn()
			if done {
				if reward == 0 {
					success.Set(1)
				}
				log.Infov("final state", observation.Data())
				log.Successf("Episode %d finished after %d timesteps, with success of %v", episode.I, timestep.I+1, success.Scalar())
				break
			}
			log.Infov("setting state", observation.Data())
			state = observation
		}
		agent.Remember(episodeEvents...)
		if success.Scalar() == 0 {
			err = agent.Hindsight(episodeEvents)
			require.NoError(err)
		}
		episode.Log()
	}
	agent.Wait()
}
