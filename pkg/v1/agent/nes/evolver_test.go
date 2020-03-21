package nes

import (
	"sync"
	"testing"

	"github.com/aunum/log"

	"github.com/stretchr/testify/require"

	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	"github.com/aunum/gold/pkg/v1/dense"
	"gorgonia.org/tensor"
)

type testBlackBox struct {
	solution      *tensor.Dense
	solvedChecker SolvedChecker
}

// Run the test black box which tries to minimize the distance between the weights and a static
// solution vector.
func (t *testBlackBox) Run(weights *tensor.Dense) (reward float32, err error) {
	cost, err := dense.BroadcastSub(t.solution, weights)
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

// RunAsync runs the black box async
func (t *testBlackBox) RunAsync(populationID int, weights *tensor.Dense, results chan BlackBoxResult, wg *sync.WaitGroup) {
	defer wg.Done()
	reward, err := t.Run(weights)
	res := BlackBoxResult{Reward: reward, Err: err, PopulationID: populationID}
	if t.solvedChecker(reward) {
		res.Solved = true
		res.Weights = weights
	}
	results <- res
}

// Init the weights for the test.
func (t *testBlackBox) InitWeights() *tensor.Dense {
	return dense.RandN(tensor.Float32, 4, 2)
}

func TestEvolver(t *testing.T) {
	solution := tensor.New(tensor.WithBacking([]float32{0.1, 0.8, -0.5, 4.0, 6.3, -2.5, -6.0, 2.7}), tensor.WithShape(4, 2))
	blackBox := &testBlackBox{solution: solution, solvedChecker: func(reward float32) bool {
		if reward > -0.001 {
			return true
		}
		return false
	}}
	config := &EvolverConfig{
		EvolverHyperparameters: DefaultEvolverHyperparameters,
		BlackBox:               blackBox,
		Base:                   agentv1.NewBase("nes", agentv1.WithLogger(log.NewLogger(log.InfoLevel, false))),
	}
	config.NGen = 10000
	evolver := NewEvolver(config)

	finalWeights, err := evolver.Evolve()
	require.NoError(t, err)
	log.Infovb("target", solution)
	log.Infovb("final weights", finalWeights)

	finalReward, err := blackBox.Run(finalWeights)
	require.NoError(t, err)

	log.Infov("final reward", finalReward)
	require.Less(t, finalReward, float32(1e-3))
}
