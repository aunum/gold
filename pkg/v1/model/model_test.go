package model_test

import (
	"testing"

	"github.com/pbarker/go-rl/pkg/v1/dense"

	"github.com/stretchr/testify/require"

	. "github.com/pbarker/go-rl/pkg/v1/model"
	l "github.com/pbarker/go-rl/pkg/v1/model/layers"
	"github.com/pbarker/log"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestSequential(t *testing.T) {
	batchSize := 10
	x := tensor.New(tensor.WithShape(batchSize, 5), tensor.WithBacking(tensor.Range(tensor.Float32, 0, 50)))
	x0, err := x.Slice(dense.MakeRangedSlice(0, 1))
	require.NoError(t, err)

	xi := NewInput("x", x0.Shape())

	y := tensor.New(tensor.WithShape(batchSize, 3), tensor.WithBacking(tensor.Range(tensor.Float32, 15, 45)))
	y0, err := y.Slice(dense.MakeRangedSlice(0, 1))
	require.NoError(t, err)

	yi := NewInput("y", y0.Shape())

	log.Infovb("x", x)
	log.Infovb("y", y)
	log.Break()

	log.Infov("x0", x0)
	log.Infov("y0", y0)
	log.Break()

	model, err := NewSequential("test")
	require.NoError(t, err)
	model.AddLayers(
		l.NewFC(5, 24, l.WithActivation(l.Sigmoid())),
		l.NewFC(24, 24, l.WithActivation(l.Sigmoid())),
		l.NewFC(24, 3, l.WithActivation(l.Linear())),
	)

	optimizer := g.NewAdamSolver()
	err = model.Compile(xi, yi,
		WithOptimizer(optimizer),
		WithLoss(MeanSquaredError),
		WithBatchSize(batchSize),
	)
	require.NoError(t, err)
	log.Break()

	prediction, err := model.Predict(x0)
	require.NoError(t, err)
	log.Infov("y0", y0)
	log.Infov("initial single prediction", prediction)
	log.Break()

	numSteps := 10000
	log.Infof("fitting for %v steps", numSteps)
	for i := 0; i < numSteps; i++ {
		err = model.FitBatch(x, y)
		require.NoError(t, err)
	}
	log.Break()
	prediction, err = model.PredictBatch(x)
	require.NoError(t, err)
	log.Infovb("y", y)
	log.Infovb("final batch prediction", prediction)
	log.Break()

	prediction, err = model.Predict(x0)
	require.NoError(t, err)
	log.Infov("y0", y0)
	log.Infov("final single prediction", prediction)

	// model.Visualize()
	// err = model.Tracker.PrintHistoryAll()
	// require.NoError(t, err)
}
