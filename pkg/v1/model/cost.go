package model

import (
	g "gorgonia.org/gorgonia"
)

// LossFn is a cost function.
type LossFn func(yHat, y *g.Node) (loss *g.Node, err error)

// MeanSquaredError cost function.
// https://en.wikipedia.org/wiki/Mean_squared_error
func MeanSquaredError(yHat, y *g.Node) (loss *g.Node, err error) {
	loss = g.Must(g.Sub(yHat, y))
	loss = g.Must(g.Square(loss))
	loss = g.Must(g.Mean(loss))
	return
}

// CrossEntropy cost function.
// https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression
func CrossEntropy(yHat, y *g.Node) (cost *g.Node, err error) {
	losses, err := g.HadamardProd(yHat, y)
	if err != nil {
		return nil, err
	}
	cost = g.Must(g.Mean(losses))
	cost = g.Must(g.Neg(cost))
	return
}

// Huber loss function.
// https://en.wikipedia.org/wiki/Huber_loss
// TODO: is this possible without conditional graphs?
// func Huber(yHat, y *g.Node) (cost *g.Node, err error) {
// 	losses := g.Must(g.Sub(y, yHat))
// 	abs := g.Must(g.Abs(losses))
// 	var absVal g.Value
// 	g.Read(abs, absVal)
// 	square := g.Must(g.Square(losses))
// }
