package model

import (
	g "gorgonia.org/gorgonia"
)

// CostFn is a cost function.
type CostFn func(prediction, y *g.Node) (cost *g.Node, err error)

// MeanSquaredError loss function.
// https://en.wikipedia.org/wiki/Mean_squared_error
func MeanSquaredError(prediction, y *g.Node) (cost *g.Node, err error) {
	losses := g.Must(g.Sub(y, prediction))
	square := g.Must(g.Square(losses))
	cost = g.Must(g.Mean(square))
	return
}

// CrossEntropy loss function.
// https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression
func CrossEntropy(prediction, y *g.Node) (cost *g.Node, err error) {
	losses, err := g.HadamardProd(prediction, y)
	if err != nil {
		return nil, err
	}
	cost = g.Must(g.Mean(losses))
	cost = g.Must(g.Neg(cost))
	return
}
