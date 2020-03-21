// Package conv2d provides a 2D convolution layer.
package conv2d

import (
	"github.com/aunum/gold/pkg/v1/model/layers"
	"github.com/aunum/gold/pkg/v1/model/layers/activation"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	t "gorgonia.org/tensor"
)

// Layer is a two dimensional convolution layer.
type Layer struct {
	Name       string
	activation activation.Fn

	input, output int
	height, width int
	filterShape   t.Shape
	kernelShape   t.Shape
	pad           []int
	stride        []int
	dilation      []int

	dtype     t.Dtype
	filter    *g.Node
	init      g.InitWFn
	shared    *Layer
	isBatched bool
}

// New returns a new Conv2D layer.
func New(input, output, height, width int, opts ...Opt) *Layer {
	filterShape := []int{output, input, height, width}
	c := &Layer{
		input:       input,
		output:      output,
		height:      height,
		width:       width,
		filterShape: filterShape,
		kernelShape: []int{3, 3},
		pad:         []int{1, 1},
		stride:      []int{1, 1},
		dilation:    []int{1, 1},
		init:        g.GlorotU(1.0),
		dtype:       t.Float32,
		activation:  activation.ReLU,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Opt is a Conv2D construction option.
type Opt func(*Layer)

// WithActivation adds an activation function to a convolution layer.
// Defaults to ReLU
// TODO: this is a bit ugly, should maybe be in its own package.
func WithActivation(fn activation.Fn) func(*Layer) {
	return func(l *Layer) {
		l.activation = fn
	}
}

// WithKernelShape sets the kernel shape.
// Defaults to (3, 3)
func WithKernelShape(kernel t.Shape) func(*Layer) {
	return func(l *Layer) {
		l.kernelShape = kernel
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

// WithDilation sets the stride size.
// Defaults to (1, 1)
func WithDilation(dilation []int) func(*Layer) {
	return func(l *Layer) {
		l.dilation = dilation
	}
}

// WithInit adds an init function to a convolution weights.
// Defaults to Glorot.
func WithInit(fn g.InitWFn) func(*Layer) {
	return func(l *Layer) {
		l.init = fn
	}
}

// WithName gives a name to a convolution.
func WithName(name string) func(*Layer) {
	return func(l *Layer) {
		l.Name = name
	}
}

// Compile the layer.
func (l *Layer) Compile(graph *g.ExprGraph, opts *layers.CompileOpts) {
	l.applyCompileOpts(opts)
	if l.shared != nil {
		l.filter = g.NewTensor(graph, l.dtype, 4, g.WithShape(l.filterShape...), g.WithInit(l.init), g.WithName(l.Name), g.WithValue(l.shared.filter.Value()))
		return
	}
	l.filter = g.NewTensor(graph, l.dtype, 4, g.WithShape(l.filterShape...), g.WithInit(l.init), g.WithName(l.Name))
}

func (l *Layer) applyCompileOpts(opts *layers.CompileOpts) {
	if opts != nil {
		if l.shared != nil {
			l.shared = opts.SharedLearnables.(*Layer)
		}
		l.isBatched = opts.AsBatch
		if (tensor.Dtype{}) != opts.AsType {
			l.dtype = opts.AsType
		}
	}
}

// Fwd is a forward pass through the layer.
func (l *Layer) Fwd(x *g.Node) (*g.Node, error) {
	n, err := g.Conv2d(x, l.filter, l.kernelShape, l.pad, l.stride, l.dilation)
	if err != nil {
		return nil, err
	}
	return n, nil
}

// Learnables returns all learnable nodes within this layer.
func (l *Layer) Learnables() g.Nodes {
	return g.Nodes{l.filter}
}

// Clone the layer.
func (l *Layer) Clone() layers.Layer {
	return &Layer{
		Name:        l.Name,
		activation:  l.activation.Clone(),
		input:       l.input,
		output:      l.output,
		height:      l.height,
		width:       l.width,
		filterShape: l.filterShape,
		kernelShape: l.kernelShape,
		pad:         l.pad,
		stride:      l.stride,
		dilation:    l.dilation,
		dtype:       l.dtype,
		init:        l.init,
		isBatched:   l.isBatched,
	}
}

// Graph returns the graph for this layer.
func (l *Layer) Graph() *g.ExprGraph {
	if l.filter == nil {
		return nil
	}
	return l.filter.Graph()
}
