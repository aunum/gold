package main

import (

	"github.com/pbarker/logger"
	"github.com/pbarker/sphere/pkg/common/errors"
	sphere "github.com/pbarker/sphere/pkg/env"
	"github.com/pbarker/go-rl/pkg/v1/agent/deepq"
)

func main() {
	s, err := sphere.NewLocalServer(sphere.GymServerConfig)
	errors.Require(err)
	defer s.Resource.Close()

	env, err := s.Make("CartPole-v0")
	errors.Require(err)

	// Get the size of the action space.
	actionSpaceSize := env.GetActionSpace().GetDiscrete().GetN()

	agent := deepq.NewAgent(deepq.DefaultAgentConfig)

	numEpisodes := 30
	logger.Infof("running for %d episodes", numEpisodes)
	for i := 0; i <= numEpisodes; i++ {
		state, err := env.Reset()
		errors.Require(err)
		// fmt.Printf("state: %v\n", state)
		// fmt.Printf("discreteState: %v\n", discreteState)

		for ts := 0; ts <= int(env.MaxEpisodeSteps); ts++ {

			// Get an action from the agent.
			action, err := agent.Action(discreteState)
			errors.Require(err)

			outcome, err := env.Step(action)
			errors.Require(err)
			discreteObv, err := obvBinner.Bin(outcome.Observation)
			// fmt.Printf("observation: %v\n", outcome.Observation)
			// fmt.Printf("dobservation: %v\n", discreteObv)

			// Learn!
			err = agent.Learn(action, outcome.Reward, discreteState, discreteObv)
			errors.Require(err)

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

