package layers

import (
	g "gorgonia.org/gorgonia"
)

// Layer in a network.
type Layer interface {
	// Compile the layer.
	Compile(x *g.Node, opts ...LayerOpt)

	// Fwd is a foward pass through the layer.
	Fwd(x *g.Node) (*g.Node, error)

	// Learnables returns all learnable nodes within this layer.
	Learnables() g.Nodes

	// Clone the layer.
	Clone() Layer
}

// LayerOpt is a layer option.
type LayerOpt func(Layer)

// WithSharedLearnables shares the learnables from another layer.
func WithSharedLearnables(shared Layer) func(Layer) {
	return func(l Layer) {
		switch lay := l.(type) {
		case *FC:
			lay.shared = shared.(*FC)
		}
	}
}

// AsBatch informs the layer compilation that it is a batch.
func AsBatch() func(Layer) {
	return func(l Layer) {
		fc := l.(*FC)
		fc.isBatched = true
	}
}
