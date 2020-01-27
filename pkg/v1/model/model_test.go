package model_test

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	. "github.com/pbarker/go-rl/pkg/v1/model"
	l "github.com/pbarker/go-rl/pkg/v1/model/layers"
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

	model, err := NewSequential("test")
	require.NoError(t, err)
	fmt.Println("adding layers")
	model.AddLayers(
		l.NewFC(12, 24, l.WithActivation(l.Sigmoid())),
		l.NewFC(24, 4, l.WithActivation(l.Sigmoid())),
	)

	optimizer := g.NewVanillaSolver(g.WithLearnRate(1.0))
	fmt.Println("compiling")
	err = model.Compile(x, y,
		WithOptimizer(optimizer),
		WithLoss(MeanSquaredError),
	)
	require.NoError(t, err)
	fmt.Println("predicting")
	fmt.Println("shape x: ", x.Shape())
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

	// model.Visualize()
	err = model.Tracker.PrintHistoryAll()
	require.NoError(t, err)
}
