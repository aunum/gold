package main

import (

	"github.com/pbarker/logger"
	"github.com/pbarker/sphere/pkg/v1/common/require"
	sphere "github.com/pbarker/sphere/pkg/env"
	"github.com/pbarker/go-rl/pkg/v1/agent/deepq"
)

func main() {
	s, err := sphere.NewLocalServer(sphere.GymServerConfig)
	require.NoError(err)
	defer s.Resource.Close()

	env, err := s.Make("CartPole-v0")
	require.NoError(err)

	// Get the size of the action space.
	actionSpaceSize := env.GetActionSpace().GetDiscrete().GetN()

	agent := deepq.NewAgent(deepq.DefaultAgentConfig)

	numEpisodes := 30
	logger.Infof("running for %d episodes", numEpisodes)
	for i := 0; i <= numEpisodes; i++ {
		state, err := env.Reset()
		require.NoError(err)
		// fmt.Printf("state: %v\n", state)
		// fmt.Printf("discreteState: %v\n", discreteState)

		for ts := 0; ts <= int(env.MaxEpisodeSteps); ts++ {

			// Get an action from the agent.
			action, err := agent.Action(discreteState)
			require.NoError(err)

			outcome, err := env.Step(action)
			require.NoError(err)

			discreteObv, err := obvBinner.Bin(outcome.Observation)
			require.NoError(err)

			// Learn!
			err = agent.Learn(action, outcome.Reward, discreteState, discreteObv)
			require.NoError(err)

			if outcome.Done {
				logger.Successf("Episode %d finished after %d timesteps", i, ts+1)
				agent.Visualize()
				// should we clear the state table here?
				break
			}
			discreteState = discreteObv
		}
	}
	res, err := env.Results()
	if err != nil {
		return nil, err
	}
	env.End()
}

