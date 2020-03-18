// Package maxpooling2d provides a max pooling 2D layer.
package maxpooling2d

import (
	"github.com/pbarker/go-rl/pkg/v1/model/layers"
	g "gorgonia.org/gorgonia"
	t "gorgonia.org/tensor"
)

// Layer implements the max pooling 2d function.
type Layer struct {
	Name string

	kernel t.Shape
	pad    []int
	stride []int

	graph *g.ExprGraph
}

// New returns a new MaxPooling2D layer.
func New(opts ...Opt) *Layer {
	c := &Layer{}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Opt is a MaxPooling2D construction option.
type Opt func(*Layer)

// WithKernelShape sets the kernel shape.
// Defaults to (2, 2)
func WithKernelShape(kernel t.Shape) func(*Layer) {
	return func(c *Layer) {
		c.kernel = kernel
	}
}

// WithPad sets the padding size.
// Defaults to (1, 1)
func WithPad(pad []int) func(*Layer) {
	return func(l *Layer) {
		l.pad = pad
	}
}

// WithStride sets the stride size.
// Defaults to (1, 1)
func WithStride(stride []int) func(*Layer) {
	return func(l *Layer) {
		l.stride = stride
	}
}

// Compile the layer.
func (l *Layer) Compile(graph *g.ExprGraph, opts *layers.CompileOpts) {
	l.graph = graph
}

// Fwd is a forward pass through the layer.
func (l *Layer) Fwd(x *g.Node) (*g.Node, error) {
	n, err := g.MaxPool2D(x, l.kernel, l.pad, l.stride)
	if err != nil {
		return nil, err
	}
	return n, nil
}

// Learnables returns all learnable nodes within this layer.
func (l *Layer) Learnables() g.Nodes {
	return g.Nodes{}
}

// Clone the layer.
func (l *Layer) Clone() layers.Layer {
	return &Layer{}
}

// Graph returns the graph for this layer.
func (l *Layer) Graph() *g.ExprGraph {
	return l.graph
}
