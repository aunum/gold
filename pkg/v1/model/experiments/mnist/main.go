package main

import (
	"fmt"

	"github.com/pbarker/go-rl/pkg/v1/common"

	"github.com/pbarker/go-rl/pkg/v1/dense"

	"github.com/pbarker/go-rl/pkg/v1/common/require"

	. "github.com/pbarker/go-rl/pkg/v1/model"
	l "github.com/pbarker/go-rl/pkg/v1/model/layers"
	"github.com/pbarker/logger"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/examples/mnist"
)

func main() {
	x, y, err := mnist.Load("train", "./testdata", g.Float32)
	require.NoError(err)

	fmt.Println("x shape: ", x.Shape())
	fmt.Println("y shape: ", y.Shape())

	exampleSize := x.Shape()[0]
	logger.Infov("exampleSize", exampleSize)

	batchSize := 100
	logger.Infov("batchsize", batchSize)
	batches := exampleSize / batchSize
	logger.Infov("batches", batches)

	x0, err := x.Slice(dense.MakeRangedSlice(0, batchSize))
	require.NoError(err)
	fmt.Println("x0: ", x0.Shape())

	y0, err := y.Slice(dense.MakeRangedSlice(0, batchSize))
	require.NoError(err)
	fmt.Println("y0: ", y0.Shape())

	model, err := NewSequential("train", x0, y0)
	require.NoError(err)

	model.AddLayers(
		l.NewFC(784, 300, l.WithActivation(l.ReLU()), l.WithInit(g.GlorotN(1)), l.WithName("w0")),
		l.NewFC(300, 100, l.WithActivation(l.ReLU()), l.WithInit(g.GlorotN(1)), l.WithName("w1")),
		l.NewFC(100, 10, l.WithActivation(l.Softmax()), l.WithInit(g.GlorotN(1)), l.WithName("w2")),
	)

	optimizer := g.NewRMSPropSolver(g.WithBatchSize(float64(batchSize)))
	err = model.Compile(
		WithOptimizer(optimizer),
		WithLoss(CrossEntropy),
		AsBatch(),
	)
	require.NoError(err)

	epochs := 100

	logger.Infov("epochs", epochs)
	for epoch := 0; epoch < epochs; epoch++ {
		for batch := 0; batch < batches; batch++ {
			start := batch * batchSize
			end := start + batchSize
			if start >= exampleSize {
				break
			}
			if end > exampleSize {
				end = exampleSize
			}
			xi, err := x.Slice(dense.MakeRangedSlice(start, end))
			require.NoError(err)
			yi, err := y.Slice(dense.MakeRangedSlice(start, end))
			require.NoError(err)

			err = model.Fit(xi, yi)
			require.NoError(err)
			// for _, layer := range model.Layers.Layers {
			// 	f := layer.(*l.FC)
			// 	// logger.Infov(fmt.Sprintf("last value %v", f.Name), f.LastValue())
			// 	for _, learnable := range layer.Learnables() {
			// 		logger.Infov(fmt.Sprintf("learnable %v", f.Name), learnable.Value())
			// 	}
			// }
		}
		loss, err := model.Tracker.GetValue("train_loss")
		require.NoError(err)
		logger.Infof("completed train epoch %v with loss %v", epoch, loss.Scalar())

	}
	// load our test set
	logger.Info("loading test set")
	x, y, err = mnist.Load("train", "./testdata", g.Float32)
	require.NoError(err)

	exampleSize = x.Shape()[0]
	losses := []float64{}
	for epoch := 0; epoch < epochs; epoch++ {
		for batch := 0; batch < batches; batch++ {
			start := batch * batchSize
			end := start + batchSize
			if start >= exampleSize {
				break
			}
			if end > exampleSize {
				end = exampleSize
			}

			xi, err := x.Slice(dense.MakeRangedSlice(start, end))
			require.NoError(err)
			yi, err := y.Slice(dense.MakeRangedSlice(start, end))
			require.NoError(err)

			err = model.Fit(xi, yi)
			require.NoError(err)
			loss, err := model.Tracker.GetValue("train_loss")
			require.NoError(err)
			losses = append(losses, loss.Scalar())
		}
		loss, err := model.Tracker.GetValue("train_loss")
		require.NoError(err)
		logger.Infof("completed eval epoch %v with loss of %v", epoch, loss.Scalar())
	}
	logger.Infov("mean eval loss", common.Mean(losses))
}
