package model

import (
	g "gorgonia.org/gorgonia"
)

// Loss is the loss of a model.
type Loss interface {
	// Comput the loss.
	Compute(yHat, y *g.Node) (loss *g.Node, err error)

	// Clone the loss to another graph.
	CloneTo(graph *g.ExprGraph, opts ...CloneOpt) Loss

	// Inputs return any inputs the loss function utilizes.
	Inputs() Inputs
}

// Reducer is used to reduce tensors to scalar values.
type Reducer func(n *g.Node, along ...int) (*g.Node, error)

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
func (m *MSELoss) CloneTo(graph *g.ExprGraph, opts ...CloneOpt) Loss {
	return m
}

// Inputs returns any inputs the loss function utilizes.
func (m *MSELoss) Inputs() Inputs {
	return Inputs{}
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
func (c *CrossEntropyLoss) CloneTo(graph *g.ExprGraph, opts ...CloneOpt) Loss {
	return c
}

// Inputs returns any inputs the loss function utilizes.
func (c *CrossEntropyLoss) Inputs() Inputs {
	return Inputs{}
}

// PseudoHuberLoss is a loss that is less sensetive to outliers.
// Can be thought of as absolute error when large, and quadratic when small.
// The larger the Delta param the steeper the loss.
type PseudoHuberLoss struct {
	// Delta determines where the function switches behavior.
	Delta float32
}

// PseudoHuber is the Huber loss function.
var PseudoHuber = &PseudoHuberLoss{
	Delta: 1.0,
}

// NewPseudoHuberLoss return a new huber loss.
func NewPseudoHuberLoss(delta float32, reducer Reducer) *PseudoHuberLoss {
	return &PseudoHuberLoss{
		Delta: delta,
	}
}

// Compute the loss.
func (h *PseudoHuberLoss) Compute(yHat, y *g.Node) (loss *g.Node, err error) {
	loss, err = g.Sub(yHat, y)
	if err != nil {
		return nil, err
	}
	loss, err = g.Div(loss, g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(float32(h.Delta))))
	if err != nil {
		return nil, err
	}
	loss, err = g.Square(loss)
	if err != nil {
		return nil, err
	}
	loss, err = g.Add(g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(float32(1.0))), loss)
	if err != nil {
		return nil, err
	}
	loss, err = g.Sqrt(loss)
	if err != nil {
		return nil, err
	}
	loss, err = g.Sub(loss, g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(float32(1.0))))
	if err != nil {
		return nil, err
	}
	deltaSquare, err := g.Square(g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(float32(h.Delta))))
	if err != nil {
		return nil, err
	}
	loss, err = g.Mul(deltaSquare, loss)
	if err != nil {
		return nil, err
	}
	return
}

// CloneTo another graph.
func (h *PseudoHuberLoss) CloneTo(graph *g.ExprGraph, opts ...CloneOpt) Loss {
	return h
}

// Inputs returns any inputs the loss function utilizes.
func (h *PseudoHuberLoss) Inputs() Inputs {
	return Inputs{}
}
