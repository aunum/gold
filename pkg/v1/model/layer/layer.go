// Package layer provides the layers for sequential models.
package layer

import (
	g "gorgonia.org/gorgonia"
	t "gorgonia.org/tensor"
)

// Config is the config for a layer.
type Config interface {
	// Compile the layer.
	Compile(graph *g.ExprGraph, opts ...CompileOpt) Layer

	// ApplyDefaults to the config.
	ApplyDefaults()

	// Clone the layer config.
	Clone() Config
}

// Layer in a network.
type Layer interface {
	// Fwd is a forward pass through the layer.
	Fwd(x *g.Node) (*g.Node, error)

	// Learnables returns all learnable nodes within this layer.
	Learnables() g.Nodes

	// Clone the layer.
	Clone() Layer

	// Graph returns the graph for this layer.
	Graph() *g.ExprGraph
}

// CompileOpt is a layer option.
type CompileOpt func(Layer)

// WithSharedLearnables shares the learnables from another layer.
func WithSharedLearnables(shared Layer) func(Layer) {
	return func(l Layer) {
		switch lay := l.(type) {
		case *fc:
			lay.shared = shared.(*fc)
		}
	}
}

// AsBatch informs the layer compilation that it is a batch.
func AsBatch() func(Layer) {
	return func(l Layer) {
		fc := l.(*fc)
		fc.isBatched = true
	}
}

// AsType sets the datatype for the layer.
func AsType(dtype t.Dtype) func(Layer) {
	return func(l Layer) {
		switch lay := l.(type) {
		case *fc:
			lay.dtype = dtype
		}
	}
}
