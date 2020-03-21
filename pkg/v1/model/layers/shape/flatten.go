// Package shape provides layers for reshaping data.
package shape

import (
	"fmt"

	"github.com/aunum/gold/pkg/v1/model/layers"
	g "gorgonia.org/gorgonia"
)

// FlattenLayer reshapes the incoming tensor to be flat, preserving the batch.
type FlattenLayer struct {
	graph *g.ExprGraph
}

// Flatten returns a new MaxPooling2D layer.
func Flatten() *FlattenLayer {
	return &FlattenLayer{}
}

// Compile the layer.
func (l *FlattenLayer) Compile(graph *g.ExprGraph, opts *layers.CompileOpts) {
	l.graph = graph
}

// Fwd is a forward pass through the layer.
func (l *FlattenLayer) Fwd(x *g.Node) (*g.Node, error) {
	if len(x.Shape()) < 2 {
		return nil, fmt.Errorf("flatten expects input in the shape (batch, x...), to few params in %v", x.Shape())
	}
	batch := x.Shape()[0]
	s := x.Shape()[1:]
	product := 1
	for _, d := range s {
		product *= d
	}
	newShape := []int{batch, product}
	n, err := g.Reshape(x, newShape)
	if err != nil {
		return nil, err
	}
	return n, nil
}

// Learnables returns all learnable nodes within this layer.
func (l *FlattenLayer) Learnables() g.Nodes {
	return g.Nodes{}
}

// Clone the layer.
func (l *FlattenLayer) Clone() layers.Layer {
	return &FlattenLayer{}
}

// Graph returns the graph for this layer.
func (l *FlattenLayer) Graph() *g.ExprGraph {
	return l.graph
}
