package main

import (
	"fmt"

	"gorgonia.org/tensor"

	"github.com/pbarker/go-rl/pkg/v1/common"

	"github.com/pbarker/go-rl/pkg/v1/dense"

	"github.com/pbarker/go-rl/pkg/v1/common/require"

	. "github.com/pbarker/go-rl/pkg/v1/model"
	l "github.com/pbarker/go-rl/pkg/v1/model/layers"
	"github.com/pbarker/log"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/examples/mnist"
)

func main() {
	x, y, err := mnist.Load("train", "./testdata", g.Float32)
	require.NoError(err)

	fmt.Println("x shape: ", x.Shape())
	fmt.Println("y shape: ", y.Shape())

	exampleSize := x.Shape()[0]
	log.Infov("exampleSize", exampleSize)

	batchSize := 100
	log.Infov("batchsize", batchSize)
	batches := exampleSize / batchSize
	log.Infov("batches", batches)

	x0, err := x.Slice(dense.MakeRangedSlice(0, 1))
	require.NoError(err)
	fmt.Println("x0: ", x0.Shape())

	y0, err := y.Slice(dense.MakeRangedSlice(0, 1))
	require.NoError(err)
	fmt.Println("y0: ", y0.Shape())

	model, err := NewSequential("mnist-solver")
	require.NoError(err)

	model.AddLayers(
		l.NewFC(784, 300, l.WithActivation(l.ReLU()), l.WithInit(g.GlorotN(1)), l.WithName("w0")),
		l.NewFC(300, 100, l.WithActivation(l.ReLU()), l.WithInit(g.GlorotN(1)), l.WithName("w1")),
		l.NewFC(100, 10, l.WithActivation(l.Softmax()), l.WithInit(g.GlorotN(1)), l.WithName("w2")),
	)

	optimizer := g.NewRMSPropSolver(g.WithBatchSize(float64(batchSize)))
	err = model.Compile(x0, y0,
		WithOptimizer(optimizer),
		WithLoss(MeanSquaredError),
		WithBatchSize(batchSize),
	)
	require.NoError(err)

	epochs := 100

	log.Infov("epochs", epochs)
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

			log.Infov("xi shape", xi.Shape())
			err = model.FitBatch(xi, yi)
			require.NoError(err)
			// for _, layer := range model.Layers.Layers {
			// 	f := layer.(*l.FC)
			// 	// log.Infov(fmt.Sprintf("last value %v", f.Name), f.LastValue())
			// 	for _, learnable := range layer.Learnables() {
			// 		log.Infov(fmt.Sprintf("learnable %v", f.Name), learnable.Value())
			// 	}
			// }
			x1, err := x.Slice(dense.MakeRangedSlice(0, 1))
			require.NoError(err)
			log.Infov("x1", x1)

			x1m := x1.Materialize().(*tensor.Dense)
			err = dense.ExpandDims(x1m, 0)
			require.NoError(err)
			log.Infov("x1m shape", x1m.Shape())
			pred, err := model.Predict(x1m)
			require.NoError(err)
			log.Infov("prediction", pred)
		}
		loss, err := model.Tracker.GetValue("train_loss")
		require.NoError(err)
		log.Infof("completed train epoch %v with loss %v", epoch, loss.Scalar())

	}
	// load our test set
	log.Info("loading test set")
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
		log.Infof("completed eval epoch %v with loss of %v", epoch, loss.Scalar())
	}
	log.Infov("mean eval loss", common.Mean(losses))
}
