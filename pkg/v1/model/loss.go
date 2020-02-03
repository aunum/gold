package model

import (
	g "gorgonia.org/gorgonia"
)

// Loss is the loss of a model.
type Loss interface {
	// Comput the loss.
	Compute(yHat, y *g.Node) (loss *g.Node, err error)

	// Clone the loss to another graph.
	CloneTo(graph *g.ExprGraph) Loss
}

// MSE is standard mean squared error loss.
var MSE = &MSELoss{}

// MSELoss is mean squared error loss.
type MSELoss struct{}

// Compute the loss
func (m *MSELoss) Compute(yHat, y *g.Node) (loss *g.Node, err error) {
	loss, err = g.Sub(yHat, y)
	if err != nil {
		return nil, err
	}
	loss, err = g.Square(loss)
	if err != nil {
		return nil, err
	}
	loss, err = g.Mean(loss)
	if err != nil {
		return nil, err
	}
	return
}

// CloneTo another graph.
func (m *MSELoss) CloneTo(graph *g.ExprGraph) Loss {
	return m
}

// CrossEntropy loss.
var CrossEntropy = &CrossEntropyLoss{}

// CrossEntropyLoss is standard cross entropy loss.
type CrossEntropyLoss struct{}

// Compute the loss.
func (c *CrossEntropyLoss) Compute(yHat, y *g.Node) (loss *g.Node, err error) {
	loss, err = g.HadamardProd(yHat, y)
	if err != nil {
		return nil, err
	}
	loss, err = g.Mean(loss)
	if err != nil {
		return nil, err
	}
	loss, err = g.Neg(loss)
	if err != nil {
		return nil, err
	}
	return
}

// CloneTo another graph.
func (c *CrossEntropyLoss) CloneTo(graph *g.ExprGraph) Loss {
	return c
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
