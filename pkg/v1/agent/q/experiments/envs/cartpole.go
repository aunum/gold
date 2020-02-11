package envs

import (
	"fmt"

	. "github.com/pbarker/go-rl/pkg/v1/agent/q"
	"github.com/pbarker/go-rl/pkg/v1/dense"
	sphere "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/pbarker/log"
	"gorgonia.org/tensor"
)

// CartPoleTestConfig is the configuration to run a cartpole test.
type CartPoleTestConfig struct {
	*Hyperparameters
	Buckets     []int
	NumEpisodes int
}

// TestCartPole tests the cartpole env.
func TestCartPole(s *sphere.Server, c CartPoleTestConfig) (*sphere.Results, error) {
	log.Infoy("config", c)
	env, err := s.Make("CartPole-v0")
	if err != nil {
		return nil, err
	}

	// Get the size of the action space.
	actionSpaceSize := env.GetActionSpace().GetDiscrete().GetN()

	// Create the Q-learning agent.
	agent := NewAgent(c.Hyperparameters, int(actionSpaceSize), nil)

	// fmt.Printf("obv space: %+v\n", env.GetObservationSpace().GetBox())

	// discretize the box space.
	box, err := env.BoxSpace()
	if err != nil {
		return nil, err
	}
	fmt.Printf("box space: %+v\n", box)

	box = reframeBox(box)
	// fmt.Printf("box space: %v \n", box)
	// per https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
	intervals := tensor.New(tensor.WithShape(box.Shape...), tensor.WithBacking(c.Buckets))
	obvBinner, err := dense.NewEqWidthBinner(intervals, box.Low, box.High)
	if err != nil {
		return nil, err
	}
	// fmt.Printf("binner: %#v\n", obvBinner)
	// fmt.Printf("widths: %#v\n", obvBinner.Widths())
	// fmt.Printf("bounds: %#v\n", obvBinner.Bounds())

	log.Infof("running for %d episodes", c.NumEpisodes)
	for i := 0; i <= c.NumEpisodes; i++ {
		state, err := env.Reset()
		if err != nil {
			return nil, err
		}
		discreteState, err := obvBinner.Bin(state)
		if err != nil {
			return nil, err
		}
		// fmt.Printf("state: %v\n", state)
		// fmt.Printf("discreteState: %v\n", discreteState)

		for ts := 0; ts <= int(env.MaxEpisodeSteps); ts++ {
			// Adapt hyperparams
			agent.Adapt(ts)

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
				log.Successf("Episode %d finished after %d timesteps", i, ts+1)
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
	return res, nil
}

// these bounds are currently set to infinite bounds, need to be reframed to be made discrete.
func reframeBox(box *sphere.BoxSpace) *sphere.BoxSpace {
	box.Low.Set(1, float32(-2.5))
	box.High.Set(1, float32(2.5))
	box.Low.Set(3, float32(-4.0))
	box.High.Set(3, float32(4.0))
	return box
}
