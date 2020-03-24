package nes

import (
	"sync"

	"github.com/aunum/gold/pkg/v1/track"

	"github.com/aunum/gold/pkg/v1/agent"
	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	"github.com/aunum/gold/pkg/v1/dense"
	"gorgonia.org/tensor"
	t "gorgonia.org/tensor"
)

// EvolverHyperparameters are the hyperparameters for the evolver.
type EvolverHyperparameters struct {
	// NPop is the population size.
	NPop int

	// NGen is the number of generations.
	NGen int

	// Sigma is the noise standard deviation.
	Sigma float32

	// Alpha is the learning rate.
	Alpha float32
}

// DefaultEvolverHyperparameters are the default hyperparams for the evolver.
var DefaultEvolverHyperparameters = &EvolverHyperparameters{
	NPop:  50,
	NGen:  300,
	Sigma: 0.1,
	Alpha: 0.001,
}

// EvolverConfig is the config for the evolver.
type EvolverConfig struct {
	// Hyperparameters for the evolver.
	*EvolverHyperparameters

	// BlackBox function to be optimized.
	BlackBox BlackBox

	// Base agent.
	Base *agent.Base
}

// Evolver of agents.
type Evolver struct {
	*EvolverHyperparameters
	*agent.Base

	blackBox  BlackBox
	rewardVal *track.TrackedScalarValue
}

// NewEvolver returns a new evolver.
func NewEvolver(c *EvolverConfig) *Evolver {
	if c.Base == nil {
		c.Base = agentv1.NewBase("nes")
	}
	rewardVal := c.Base.Tracker.TrackValue("reward", 0)
	return &Evolver{
		Base:                   c.Base,
		EvolverHyperparameters: c.EvolverHyperparameters,
		blackBox:               c.BlackBox,
		rewardVal:              rewardVal.(*track.TrackedScalarValue),
	}
}

// Evolve the agents.
func (e *Evolver) Evolve() (weights *tensor.Dense, err error) {
	// initialize random weights.
	weights = e.blackBox.InitWeights()

	// ensure weights are shaped to be one of many.
	err = dense.OneOfMany(weights)
	if err != nil {
		return nil, err
	}

	// Normalize static values to weight shape.
	sigmaR := dense.Fill(e.Sigma, weights.Shape()...)
	alphaR := dense.Fill(e.Alpha, weights.Shape()...)
	npopR := dense.Fill(float32(e.NPop), weights.Shape()...)

	// evolve
	e.Logger.Infof("running for %v generations", e.NGen)
	for gen := 0; gen <= e.NGen; gen++ {
		e.Logger.Infof("running generation %v with %v populations", gen, e.NPop)
		var noise *t.Dense
		rewards := t.Ones(t.Float32, e.NPop, 1)

		results := make(chan BlackBoxResult)
		var wg sync.WaitGroup
		for pop := 0; pop < e.NPop; pop++ {
			// mutate
			noisePop := dense.RandN(t.Float32, weights.Shape()...)
			noise, err = dense.ConcatOr(0, noise, noisePop)
			if err != nil {
				return nil, err
			}
			mutated, err := noisePop.Mul(sigmaR)
			if err != nil {
				return nil, err
			}
			mutated, err = weights.Add(mutated)
			if err != nil {
				return nil, err
			}
			// test
			wg.Add(1)
			go e.blackBox.RunAsync(pop, mutated, results, &wg)

		}
		// collect results
		var solved bool
		go func() {
			for result := range results {
				if result.Err != nil {
					e.Logger.Fatal(result.Err)
				}
				rewards.Set(result.PopulationID, float32(result.Reward))
				if result.Solved {
					weights = result.Weights
					solved = true
				}
			}
		}()
		wg.Wait()
		close(results)
		if solved {
			e.Logger.Successf("solved!")
			return
		}

		avgReward, err := dense.Mean(rewards)
		if err != nil {
			return nil, err
		}
		e.Logger.Infof("average reward for generation %v was %v", gen, avgReward)
		e.rewardVal.Set(avgReward.Data().(float32))
		e.Tracker.LogStep(gen, 0)

		// standardize rewards
		rewards, err = dense.ZNorm(rewards, 0)
		if err != nil {
			return nil, err
		}
		err = noise.T()
		if err != nil {
			return nil, err
		}

		// perform update to weights.
		// weights = weights + alpha/(npop*sigma) * dot(noise, rewards)
		rn, err := noise.TensorMul(rewards, []int{noise.Dims() - 1}, []int{0})
		if err != nil {
			return nil, err
		}
		err = rn.T()
		if err != nil {
			return nil, err
		}
		w, err := sigmaR.Mul(npopR)
		if err != nil {
			return nil, err
		}
		w, err = dense.Div(alphaR, w)
		if err != nil {
			return nil, err
		}
		w, err = w.Mul(rn)
		if err != nil {
			return nil, err
		}
		weights, err = weights.Add(w)
		if err != nil {
			return nil, err
		}
	}
	return
}
