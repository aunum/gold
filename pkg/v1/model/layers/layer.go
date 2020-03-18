// Package layers provides the layers for sequential models.
package layers

import (
	g "gorgonia.org/gorgonia"
	t "gorgonia.org/tensor"
)

// Layer in a network.
type Layer interface {
	// Compile the layer to the graph.
	Compile(graph *g.ExprGraph, opts *CompileOpts)

	// Fwd is a forward pass through the layer.
	Fwd(x *g.Node) (*g.Node, error)

	// Learnables returns all learnable nodes within this layer.
	Learnables() g.Nodes

	// Clone the layer.
	Clone() Layer

	// Graph returns the graph for this layer.
	Graph() *g.ExprGraph
}

// CompileOpts are general compile options.
type CompileOpts struct {
	// SharedLearnables shares the layers learnables with another layer.
	SharedLearnables Layer

	// AsBatch makes the layer a batch layer.
	AsBatch bool

	// AsType sets the data type for the layer.
	AsType t.Dtype
}
