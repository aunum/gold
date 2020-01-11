package main

import (
	"github.com/pbarker/logger"

	"github.com/pbarker/sphere/pkg/common/errors"
	"github.com/pbarker/sphere/pkg/env"
)

func main() {
	// Spin up a sphere docker container backed by an OpenAI gym server.
	server, err := env.NewLocalServer(env.GymServerConfig)
	errors.Require(err)

	// Create the a new environment.
	env, err := server.Make("CartPole-v0")
	errors.Require(err)
	env.Print()

	numEpisodes := 20
	logger.Infof("running for %d episodes", numEpisodes)
	for i := 0; i <= numEpisodes; i++ {
		// Reset environment on each episode.
		_, err := env.Reset()
		errors.Require(err)

		// Iterate thorugh the maximum number of timesteps environment allows or
		// until a 'done' state is reached
		for ts := 0; ts <= int(env.MaxEpisodeSteps); ts++ {
			action, err := env.SampleAction()
			errors.Require(err)

			// Take a step in the environment.
			resp, err := env.Step(action)
			errors.Require(err)

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
