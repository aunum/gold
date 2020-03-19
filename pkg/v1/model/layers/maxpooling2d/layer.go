// Package maxpooling2d provides a max pooling 2D layer.
package maxpooling2d

import (
	"fmt"

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
	c := &Layer{
		kernel: []int{2, 2},
		pad:    []int{0, 0},
		stride: []int{2, 2},
	}
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
// Defaults to (0, 0)
func WithPad(pad []int) func(*Layer) {
	return func(l *Layer) {
		l.pad = pad
	}
}

// WithStride sets the stride size.
// Defaults to (2, 2)
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
	fmt.Println("---- pool")
	fmt.Println("pool x shape: ", x.Shape())
	fmt.Println("kernel: ", l.kernel)
	fmt.Println("pad: ", l.pad)
	fmt.Println("stride: ", l.stride)
	n, err := g.MaxPool2D(x, l.kernel, l.pad, l.stride)
	if err != nil {
		return nil, err
	}
	fmt.Println("returning pool shape: ", n.Shape())
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
