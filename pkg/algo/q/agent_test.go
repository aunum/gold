package q_test

import (
	"fmt"
	"testing"

	. "github.com/pbarker/go-rl/pkg/algo/q"
	"github.com/pbarker/go-rl/pkg/common"
	"github.com/pbarker/sphere/pkg/common/logger"
	sphere "github.com/pbarker/sphere/pkg/env"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestAgent(t *testing.T) {
	s, err := sphere.NewLocalServer(sphere.GymServerConfig)
	require.Nil(t, err)
	defer s.Resource.Close()

	env, err := s.Make("CartPole-v0")
	require.Nil(t, err)

	// Get the size of the action space.
	actionSpaceSize := env.GetActionSpace().GetDiscrete().GetN()

	// Create the Q-learning agent.
	agent := NewAgent(DefaultHyperparameters, int(actionSpaceSize), nil)

	fmt.Printf("space: %+v\n", env.GetObservationSpace().GetBox())

	// discretize the box space.
	box, err := env.BoxSpace()
	box = reframeBox(box)
	fmt.Printf("box space: %v \n", box)
	require.Nil(t, err)
	// per https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
	intervals := tensor.New(tensor.WithShape(box.Shape...), tensor.WithBacking([]int{1, 1, 6, 12}))
	obvBinner, err := common.NewDenseEqWidthBinner(intervals, box.Low, box.High)
	fmt.Printf("binner: %#v\n", obvBinner)
	require.Nil(t, err)
	fmt.Printf("widths: %#v\n", obvBinner.Widths())
	fmt.Printf("bounds: %#v\n", obvBinner.Bounds())

	numEpisodes := 50
	logger.Infof("running for %d episodes", numEpisodes)
	for i := 0; i <= numEpisodes; i++ {
		state, err := env.Reset()
		discreteState, err := obvBinner.Bin(state)
		require.Nil(t, err)
		fmt.Printf("state: %v\n", state)
		fmt.Printf("discreteState: %v\n", discreteState)

		for ts := 0; ts <= int(env.MaxEpisodeSteps); ts++ {
			// Get an action from the agent.
			action, err := agent.Action(discreteState)
			require.Nil(t, err)

			outcome, err := env.Step(action)
			require.Nil(t, err)
			discreteObv, err := obvBinner.Bin(outcome.Observation)
			fmt.Printf("observation: %v\n", outcome.Observation)
			fmt.Printf("dobservation: %v\n", discreteObv)

			// Learn!
			err = agent.Learn(action, outcome.Reward, discreteState, discreteObv)
			require.Nil(t, err)

			if outcome.Done {
				logger.Successf("Episode %d finished after %d timesteps", i, ts+1)
				agent.Visualize()
				// should we clear the state table here?
				break
			}
		}
	}
	env.End()

	env.PlayAll()
}

// these bounds are currently set to infinite bounds, need to be reframed to be made discrete.
func reframeBox(box *sphere.BoxSpace) *sphere.BoxSpace {
	box.Low.Set(1, float32(-2.5))
	box.High.Set(1, float32(2.5))
	box.Low.Set(3, float32(-4.0))
	box.High.Set(3, float32(4.0))
	return box
}
