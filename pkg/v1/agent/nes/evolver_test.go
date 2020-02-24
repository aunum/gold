package nes

import (
	"testing"

	"github.com/pbarker/log"

	"github.com/stretchr/testify/require"

	"github.com/pbarker/go-rl/pkg/v1/dense"
	"gorgonia.org/tensor"
)

type testBlackBox struct {
	solution *tensor.Dense
}

// Run the test black box which tries to minimize the distance between the weights and a static
// solution vector.
func (t *testBlackBox) Run(weights *tensor.Dense) (reward float32, err error) {
	cost, err := t.solution.Sub(weights)
	if err != nil {
		return reward, err
	}
	squared, err := tensor.Square(cost)
	if err != nil {
		return reward, err
	}
	sum, err := tensor.Sum(squared)
	if err != nil {
		return reward, err
	}
	r, err := tensor.Neg(sum)
	if err != nil {
		return reward, err
	}
	reward = r.Data().(float32)
	return
}

// Init the weights for the test.
func (t *testBlackBox) InitWeights() *tensor.Dense {
	return dense.RandN(tensor.Float32, 3)
}

func TestEvolver(t *testing.T) {
	solution := tensor.New(tensor.WithBacking([]float32{0.1, 0.8, -0.5}))
	blackBox := &testBlackBox{solution: solution}
	config := &EvolverConfig{
		EvolverHyperparameters: DefaultEvolverHyperparameters,
		BlackBox:               blackBox,
	}
	log.Infof("running evolver for %v generations with a population size of %v", config.NGen, config.NPop)
	evolver := NewEvolver(config)

	finalWeights, err := evolver.Evolve()
	require.NoError(t, err)
	log.Infov("target", solution)
	log.Infov("final weights", finalWeights)

	finalReward, err := blackBox.Run(finalWeights)
	require.NoError(t, err)

	log.Infov("final reward", finalReward)
	require.Less(t, finalReward, float32(1e-3))
}
