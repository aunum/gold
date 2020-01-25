package layers

import (
	"gorgonia.org/tensor"

	"github.com/pbarker/go-rl/pkg/v1/model"
	g "gorgonia.org/gorgonia"
)

// FC is a fully connected layer of neurons.
type FC struct {
	// Input is the number of units in input.
	Input int

	// Output is the number of units in the output.
	Output int

	// Name of the layer.
	Name string

	activation Activation
	weights    *g.Node
	init       g.InitWFn
	dtype      tensor.Dtype
	bias       *g.Node
	useBias    bool
	biasInit   g.InitWFn
	isBatched  bool
}

// FCOpts are options for a fully connected layer.
type FCOpts func(*FC)

// NewFC returns a new fully connected layer.
func NewFC(input, output int, opts ...FCOpts) *FC {
	fc := &FC{
		Input:    input,
		Output:   output,
		dtype:    g.Float32,
		init:     g.GlorotU(1.0),
		useBias:  true,
		biasInit: g.Zeroes(),
	}
	for _, opt := range opts {
		opt(fc)
	}
	return fc
}

// WithActivation adds an activation function to the layer.
func WithActivation(fn Activation) func(*FC) {
	return func(f *FC) {
		f.activation = fn
	}
}

// WithName gives a name to this layer.
func WithName(name string) func(*FC) {
	return func(f *FC) {
		f.Name = name
	}
}

// WithInit adds an init function to the weights.
// Defaults to Glorot.
func WithInit(fn g.InitWFn) func(*FC) {
	return func(f *FC) {
		f.init = fn
	}
}

// WithType sets the type for the layer
// Defaults to Float32.
func WithType(t tensor.Dtype) func(*FC) {
	return func(f *FC) {
		f.dtype = t
	}
}

// WithBiasInit adds an init function to the bias.
// Defaults to zeros.
func WithBiasInit(fn g.InitWFn) func(*FC) {
	return func(f *FC) {
		f.init = fn
	}
}

// WithNoBias removes the bias.
func WithNoBias() func(*FC) {
	return func(f *FC) {
		f.useBias = false
	}
}

// Compile the layer into the model.
func (f *FC) Compile(model model.Model) {
	f.isBatched = model.IsBatch()
	f.weights = g.NewMatrix(model.Graph(), f.dtype, g.WithShape(f.Input, f.Output), g.WithInit(f.init), g.WithName(f.Name))
	if f.useBias {
		if model.IsBatch() {
			f.bias = g.NewMatrix(model.Graph(), f.dtype, g.WithShape(1, f.Output), g.WithInit(f.biasInit))
		} else {
			f.bias = g.NewMatrix(model.Graph(), f.dtype, g.WithShape(model.X().Shape()[0], f.Output), g.WithInit(f.biasInit))
		}
	}
}

// Fwd is a foward pass on a single fully connected layer.
// func (f *FC) Fwd(x *g.Node) (*g.Node, error) {
// 	prod := g.Must(g.Mul(x, f.weights))
// 	wb := prod
// 	if f.bias != nil {
// 		wb = g.Must(g.Add(prod, f.bias))
// 	}

// 	if f.activation == nil {
// 		return wb, nil
// 	}
// 	a := g.Must(f.activation(wb))
// 	return a, nil
// }

// Fwd is a foward pass on a single fully connected layer.
func (f *FC) Fwd(x *g.Node) (*g.Node, error) {
	var xw, xwb *g.Node
	var err error
	if xw, err = g.Mul(x, f.weights); err != nil {
		return nil, err
	}

	if f.bias == nil {
		xwb = xw
		goto act
	}

	if f.isBatched {
		if xwb, err = g.BroadcastAdd(xw, f.bias, nil, []byte{0}); err != nil {
			return nil, err
		}
	} else {
		if xwb, err = g.Add(xw, f.bias); err != nil {
			return nil, err
		}
	}
act:
	if f.activation == nil {
		return xwb, nil
	}
	a, err := f.activation.Fwd(xwb)
	return a, err
}

// Learnables are the learnable parameters of the fully connected layer.
func (f *FC) Learnables() g.Nodes {
	if f.bias != nil {
		return g.Nodes{f.weights, f.bias}
	}
	return g.Nodes{f.weights}
}
