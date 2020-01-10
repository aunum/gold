package model

import (
	g "gorgonia.org/gorgonia"
)

// CostFn is a cost function.
type CostFn func(expected, prediction *g.Node) (cost *g.Node)

// MeanSquaredError gives the mean squared error of the expected and prediction nodes.
func MeanSquaredError(expected, prediction *g.Node) (cost *g.Node) {
	return g.Must(g.Mean(g.Must(g.Square(g.Must(g.Sub(expected, prediction))))))
}