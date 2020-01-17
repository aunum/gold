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
	xB := []float32{0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1}
	x := tensor.New(tensor.WithBacking(xB), tensor.WithShape(1, 12))

	yB := []float32{0.34, 0, 5.3, 1}
	y := tensor.New(tensor.WithBacking(yB), tensor.WithShape(1, 4))

	logger.Info("x: ", x)
	logger.Info("y: ", y)

	model := NewSequential(x, y)
	model.AddLayers(
		layers.NewFC(12, 24, layers.WithActivation(g.Sigmoid)),
		layers.NewFC(24, 4, layers.WithActivation(g.Sigmoid)),
	)
	err := model.Compile(WithOptimizer(g.NewVanillaSolver(g.WithLearnRate(1.0))), WithLoss(MeanSquaredError))
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
