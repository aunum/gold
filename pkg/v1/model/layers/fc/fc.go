// Package fc provides a fully connected layer.
package fc

import (
	"fmt"

	"github.com/aunum/log"
	g "gorgonia.org/gorgonia"
	t "gorgonia.org/tensor"
)

// Layer is a fully connected layer of neurons.
type Layer struct {
	// Input is the number of units in input.
	Input int

	// Output is the number of units in the output.
	Output int

	// Name of the layer.
	Name string

	activation activation.Fn
	weights    *g.Node
	init       g.InitWFn
	dtype      t.Dtype
	bias       *g.Node
	useBias    bool
	biasInit   g.InitWFn
	isBatched  bool
	shared     *Layer
}

// Opts are options for a fully connected layer.
type Opts func(*Layer)

// New returns a new fully connected layer.
func New(input, output int, opts ...Opts) *Layer {
	fc := &Layer{
		Input:      input,
		Output:     output,
		dtype:      g.Float32,
		init:       g.GlorotU(1.0),
		useBias:    true,
		biasInit:   g.Zeroes(),
		activation: activation.ReLU,
	}
	for _, opt := range opts {
		opt(fc)
	}
	return fc
}

// WithActivation adds an activation function to a layer.
// Defaults to ReLU.
func WithActivation(fn activation.Fn) func(*Layer) {
	return func(l *Layer) {
		l.activation = fn
	}
}

// WithName gives a name to a FC layer.
func WithName(name string) func(*Layer) {
	return func(l *Layer) {
		l.Name = name
	}
}

// WithInit adds an init function to a FC weights.
// Defaults to Glorot.
func WithInit(fn g.InitWFn) func(*Layer) {
	return func(l *Layer) {
		l.init = fn
	}
}

// WithBiasInit adds an init function to a FC bias.
// Defaults to zeros.
func WithBiasInit(fn g.InitWFn) func(*Layer) {
	return func(l *Layer) {
		l.init = fn
	}
}

// WithoutBias removes a FC bias.
func WithoutBias() func(*Layer) {
	return func(l *Layer) {
		l.useBias = false
	}
}

// Compile the layer into the graph.
func (l *Layer) Compile(graph *g.ExprGraph, opts *layers.CompileOpts) {
	l.applyCompileOpts(opts)
	if l.shared != nil {
		l.weights = g.NewMatrix(graph, l.dtype, g.WithShape(l.Input, l.Output), g.WithName(l.Name), g.WithValue(l.shared.weights.Value()))
		if l.useBias {
			l.bias = g.NewMatrix(graph, l.dtype, g.WithShape(1, l.Output), g.WithName(fmt.Sprintf("%s-bias", l.Name)), g.WithValue(l.shared.bias.Value()))
		}
		return
	}
	l.weights = g.NewMatrix(graph, l.dtype, g.WithShape(l.Input, l.Output), g.WithInit(l.init), g.WithName(l.Name))
	if l.useBias {
		l.bias = g.NewMatrix(graph, l.dtype, g.WithShape(1, l.Output), g.WithInit(l.biasInit), g.WithName(fmt.Sprintf("%s-bias", l.Name)))
	}
}

func (l *Layer) applyCompileOpts(opts *layers.CompileOpts) {
	if opts != nil {
		if l.shared != nil {
			l.shared = opts.SharedLearnables.(*Layer)
		}
		l.isBatched = opts.AsBatch
		l.dtype = opts.AsType
	}
}

// Fwd is a forward pass on a single fully connected layer.
func (l *Layer) Fwd(x *g.Node) (*g.Node, error) {
	var xw, xwb *g.Node
	var err error
	if x.IsVector() {
		s := t.Shape{1}
		s = append(s, x.Shape()...)
		x, err = g.Reshape(x, s)
		if err != nil {
			return nil, err
		}
		log.Debugf("normalizing dimensions of x to %v", s)
	}

	// Note: parts of this are borrowed from https://github.com/gorgonia/golgi
	if xw, err = g.Mul(x, l.weights); err != nil {
		return nil, err
	}

	if l.bias == nil {
		xwb = xw
		goto act
	}
	if l.isBatched {
		if xwb, err = g.BroadcastAdd(xw, l.bias, nil, []byte{0}); err != nil {
			return nil, err
		}
	} else {
		if xwb, err = g.Add(xw, l.bias); err != nil {
			return nil, err
		}
	}

act:
	if l.activation == nil {
		return xwb, nil
	}
	a, err := l.activation.Fwd(xwb)
	if err != nil {
		return nil, err
	}
	return a, err
}

// Learnables are the learnable parameters of the fully connected layer.
func (l *Layer) Learnables() g.Nodes {
	if l.bias != nil {
		return g.Nodes{l.weights, l.bias}
	}
	return g.Nodes{l.weights}
}

// Clone the layer without any nodes. (nodes cannot be shared)
func (l *Layer) Clone() layers.Layer {
	return &Layer{
		Input:      l.Input,
		Output:     l.Output,
		Name:       l.Name,
		activation: l.activation.Clone(),
		init:       l.init,
		dtype:      l.dtype,
		useBias:    l.useBias,
		biasInit:   l.biasInit,
		isBatched:  l.isBatched,
	}
}

// Graph returns the graph this layer was compiled with.
func (l *Layer) Graph() *g.ExprGraph {
	if l.weights == nil {
		return nil
	}
	return l.weights.Graph()
}
