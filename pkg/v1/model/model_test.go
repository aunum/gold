package model_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	. "github.com/pbarker/go-rl/pkg/v1/model"
	"github.com/pbarker/go-rl/pkg/v1/model/layers"
	"github.com/pbarker/logger"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestSequential(t *testing.T) {
	xB := []float32{-20, 0, 10, 0, 1, 10000, 1, 0, 20, 1, 5000, -20000}
	x := tensor.New(tensor.WithBacking(xB), tensor.WithShape(1, 12))

	yB := []float32{1, 0, 0, 1}
	y := tensor.New(tensor.WithBacking(yB), tensor.WithShape(1, 4))

	logger.Infov("x", x)
	logger.Infov("y", y)

	model, err := NewSequential(x, y)
	require.NoError(t, err)
	model.AddLayers(
		layers.NewFC(12, 24, layers.WithActivation(g.Sigmoid)),
		layers.NewFC(24, 4, layers.WithActivation(g.Sigmoid)),
	)

	optimizer := g.NewVanillaSolver(g.WithLearnRate(1.0))
	err = model.Compile(WithOptimizer(optimizer), WithLoss(MeanSquaredError))
	require.NoError(t, err)

	prediction, err := model.Predict(x)
	require.NoError(t, err)
	logger.Infov("prediction", prediction)

	for i := 0; i < 10000; i++ {
		err = model.Fit(x, y)
		require.NoError(t, err)
	}
	prediction, err = model.Predict(x)
	require.NoError(t, err)
	logger.Infov("y", y)
	logger.Infov("prediction", prediction)

	model.Visualize()
}
