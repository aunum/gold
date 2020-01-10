package deepq_test

import (
	"testing"

	"github.com/pbarker/sphere/pkg/common/logger"
	sphere "github.com/pbarker/sphere/pkg/env"
	"github.com/stretchr/testify/require"
)

func TestAgent(t *testing.T) {
	s, err := sphere.NewLocalServer(sphere.GymServerConfig)
	require.Nil(t, err)
	defer s.Resource.Close()

	env, err := s.Make("CartPole-v0")
	require.Nil(t, err)

	// Get the size of the action space.
	actionSpaceSize := env.GetActionSpace().GetDiscrete().GetN()

	numEpisodes := 30
	logger.Infof("running for %d episodes", numEpisodes)
	for i := 0; i <= numEpisodes; i++ {
		state, err := env.Reset()
		require.Nil(t, err)
		// fmt.Printf("state: %v\n", state)
		// fmt.Printf("discreteState: %v\n", discreteState)

		for ts := 0; ts <= int(env.MaxEpisodeSteps); ts++ {

			// Get an action from the agent.
			action, err := agent.Action(discreteState)
			if err != nil {
				return nil, err
			}

			outcome, err := env.Step(action)
			if err != nil {
				return nil, err
			}
			discreteObv, err := obvBinner.Bin(outcome.Observation)
			// fmt.Printf("observation: %v\n", outcome.Observation)
			// fmt.Printf("dobservation: %v\n", discreteObv)

			// Learn!
			err = agent.Learn(action, outcome.Reward, discreteState, discreteObv)
			if err != nil {
				return nil, err
			}

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
