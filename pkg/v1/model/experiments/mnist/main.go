package main

import (
	"fmt"

	"github.com/aunum/gold/pkg/v1/dense"

	"github.com/aunum/gold/pkg/v1/common/require"

	"github.com/aunum/gold/pkg/v1/common/num"
	. "github.com/aunum/gold/pkg/v1/model"
	"github.com/aunum/log"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/examples/mnist"
	"gorgonia.org/tensor"
)

func main() {
	x, y, err := mnist.Load("train", "./testdata", g.Float32)
	require.NoError(err)

	// Reshape for convolution
	numExamples := x.Shape()[0]
	x.Reshape(numExamples, 1, 28, 28)
	y.Reshape(numExamples, 1, 28, 28)

	// load our test set
	testX, testY, err := mnist.Load("test", "./testdata", g.Float32)
	require.NoError(err)

	log.Infov("x batch shape", x.Shape())
	log.Infov("y batch shape", y.Shape())

	exampleSize := x.Shape()[0]
	log.Infov("exampleSize", exampleSize)

	batchSize := 100
	log.Infov("batchsize", batchSize)
	batches := exampleSize / batchSize
	log.Infov("num batches", batches)

	xi := NewInput("x", []int{1, 1, 28, 28})
	log.Infov("x input shape", xi.Shape())

	yi := NewInput("y", []int{1, 10})
	log.Infov("y input shape", yi.Shape())

	model, err := NewSequential("mnist")
	require.NoError(err)

	model.AddLayers(
		conv2d.New(1, 32, 3, 3, conv2d.WithName("conv0")),
		maxpooling2d.New(),
		conv2d.New(32, 64, 3, 3, conv2d.WithName("conv1")),
		maxpooling2d.New(),
		conv2d.New(64, 128, 3, 3, conv2d.WithName("conv3")),
		maxpooling2d.New(),
		shape.Flatten(),
		fc.New(128*3*3, 100, fc.WithInit(g.GlorotN(1)), fc.WithName("fc0")),
		fc.New(100, 10, fc.WithActivation(activation.Softmax), fc.WithInit(g.GlorotN(1)), fc.WithName("dist")),
	)

	optimizer := g.NewRMSPropSolver()
	err = model.Compile(xi, yi,
		WithOptimizer(optimizer),
		WithLoss(PseudoCrossEntropy),
		WithBatchSize(batchSize),
	)
	require.NoError(err)

	epochs := 20

	log.Infov("epochs", epochs)
	for epoch := 0; epoch < epochs; epoch++ {

		var xi tensor.View
		var yi tensor.View
		for batch := 0; batch < batches; batch++ {
			start := batch * batchSize
			end := start + batchSize
			if start >= exampleSize {
				break
			}
			if end > exampleSize {
				end = exampleSize
			}

			xi, err = x.Slice(dense.MakeRangedSlice(start, end))
			require.NoError(err)
			fmt.Println("xi shape: ", xi.Shape())
			yi, err = y.Slice(dense.MakeRangedSlice(start, end))
			require.NoError(err)

			err = model.FitBatch(xi, yi)
			require.NoError(err)
			model.Tracker.LogStep(epoch, batch)
		}
		accuracy, loss, err := evaluate(testX.(*tensor.Dense), testY.(*tensor.Dense), model, batchSize)
		require.NoError(err)
		log.Infof("completed train epoch %v with accuracy %v and loss %v", epoch, accuracy, loss)
	}
	err = model.Tracker.Clear()
	require.NoError(err)
}

func evaluate(x, y *tensor.Dense, model *Sequential, batchSize int) (acc, loss float32, err error) {
	exampleSize := x.Shape()[0]
	batches := exampleSize / batchSize

	accuracies := []float32{}
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

		yHat, err := model.PredictBatch(xi)
		require.NoError(err)

		acc, err := accuracy(yHat.(*tensor.Dense), yi.(*tensor.Dense), model)
		require.NoError(err)
		accuracies = append(accuracies, acc)
	}
	lossVal, err := model.Tracker.GetValue("mnist_train_batch_loss")
	require.NoError(err)
	loss = float32(lossVal.Scalar())
	acc = num.Mean(accuracies)
	return
}

func accuracy(yHat, y *tensor.Dense, model Model) (float32, error) {
	yMax, err := y.Argmax(1)
	require.NoError(err)

	yHatMax, err := yHat.Argmax(1)
	require.NoError(err)

	eq, err := tensor.ElEq(yMax, yHatMax, tensor.AsSameType())
	require.NoError(err)
	eqd := eq.(*tensor.Dense)
	len := eqd.Len()

	numTrue, err := eqd.Sum()
	if err != nil {
		return 0, err
	}

	return float32(numTrue.Data().(int)) / float32(len), nil
}
