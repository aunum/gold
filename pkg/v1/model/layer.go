package model

import (
	g "gorgonia.org/gorgonia"
)

// Layer in a network.
type Layer interface {
	// Fwd is a foward pass through the layer.
	Fwd(x *g.Node) (*g.Node, error)

	// Learnables returns all learnable nodes within this layer.
	Learnables() g.Nodes

	// Compile the layer.
	Compile(model Model)
}
