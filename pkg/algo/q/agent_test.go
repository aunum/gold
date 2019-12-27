package q_test

import (
	"testing"

	. "github.com/pbarker/go-rl/pkg/algo/q"
	"github.com/pbarker/sphere/pkg/common/logger"
	sphere "github.com/pbarker/sphere/pkg/env"
	"github.com/stretchr/testify/require"
)

func TestAgent(t *testing.T) {
	// Test Cartpole-v0
	s, err := sphere.NewLocalServer(sphere.GymServerConfig)
	require.Nil(t, err)

	env, err := s.Make("CartPole-v0")
	require.Nil(t, err)

	// Get the size of the action space.
	actionSpaceSize := env.GetActionSpace().GetDiscrete().GetN()

	// Create the q learning agent.
	agent := NewAgent(DefaultHyperparameters, int(actionSpaceSize), nil)

	numEpisodes := 20
	logger.Infof("running for %d episodes", numEpisodes)
	for i := 0; i <= numEpisodes; i++ {
		// Reset environment on each episode.
		_, err := env.Reset()
		require.Nil(t, err)

		// Iterate thorugh the maximum number of timesteps environment allows or
		// until a 'done' state is reached
		for ts := 0; ts <= int(env.MaxEpisodeSteps); ts++ {
			action, err := env.SampleAction()
			require.Nil(t, err)

			// Take a step in the environment.
			resp, err := env.Step(action)
			require.Nil(t, err)

			// Check if environment is complete.
			if resp.Done {
				logger.Successf("Episode %d finished after %d timesteps", i, ts+1)
				break
			}
		}
	}
	// Print results, download all videos to temp dir, and remove environment from backend.
	env.End()

	// Play all recordings.
	env.PlayAll()
}
