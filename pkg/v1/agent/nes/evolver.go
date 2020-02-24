package nes

import (
	"github.com/pbarker/go-rl/pkg/v1/dense"
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
	*EvolverHyperparameters

	// BlackBox function to be optimized.
	BlackBox BlackBox
}

// Evolver of agents.
type Evolver struct {
	*EvolverHyperparameters

	blackBox BlackBox
	sigma    *t.Dense
	npop     *t.Dense
	alpha    *t.Dense
}

// BlackBox function we wish to optimize.
type BlackBox interface {
	// Run the black box.
	Run(wieghts *t.Dense) (reward float32, err error)

	// Initialize the weights.
	InitWeights() *t.Dense
}

// NewEvolver returns a new evolver.
func NewEvolver(c *EvolverConfig) *Evolver {
	sigma := t.New(t.FromScalar(c.Sigma))
	nPop := t.New(t.FromScalar(float32(c.NPop)))
	alpha := t.New(t.FromScalar(c.Alpha))
	return &Evolver{
		EvolverHyperparameters: c.EvolverHyperparameters,
		blackBox:               c.BlackBox,
		sigma:                  sigma,
		npop:                   nPop,
		alpha:                  alpha,
	}
}

// Evolve the agents.
func (e *Evolver) Evolve() (weights *tensor.Dense, err error) {
	// initialize random weights
	weights = e.blackBox.InitWeights()

	// Normalize static values to weight shape.
	sigmaR, err := dense.Repeat(e.sigma, 0, weights.Shape()...)
	if err != nil {
		return nil, err
	}
	alphaR, err := dense.Repeat(e.alpha, 0, weights.Shape()...)
	if err != nil {
		return nil, err
	}
	npopR, err := dense.Repeat(e.npop, 0, weights.Shape()...)
	if err != nil {
		return nil, err
	}
	err = dense.OneOfMany(weights)
	if err != nil {
		return nil, err
	}

	for gen := 0; gen <= e.NGen; gen++ {
		var noise *t.Dense
		rewards := t.Ones(t.Float32, e.NPop, 1)

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
			reward, err := e.blackBox.Run(mutated)
			if err != nil {
				return nil, err
			}
			rewards.Set(pop, float32(reward))
		}
		// standardize rewards
		rewards, err = dense.ZNorm(rewards, 0)
		if err != nil {
			return nil, err
		}
		err := noise.T()
		if err != nil {
			return nil, err
		}
		// perform update to wieghts.
		rn, err := noise.MatMul(rewards)
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
