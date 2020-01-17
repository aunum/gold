package layers

import (
	"fmt"

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

	activation model.ActivationFn
	weights    *g.Node
	init       g.InitWFn
	dtype      tensor.Dtype
}

// FCOpts are options for a fully connected layer.
type FCOpts func(*FC)

// NewFC returns a new fully connected layer.
func NewFC(input, output int, opts ...FCOpts) *FC {
	fc := &FC{
		Input:  input,
		Output: output,
		dtype:  g.Float32,
		init:   g.GlorotU(1.0),
	}
	for _, opt := range opts {
		opt(fc)
	}
	return fc
}

// WithActivation adds an activation function to the layer.
func WithActivation(fn model.ActivationFn) func(*FC) {
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

// WithInit adds an init function to this layer.
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

// Compile the layer into the model.
func (f *FC) Compile(model model.Model) {
	f.weights = g.NewMatrix(model.Graph(), f.dtype, g.WithShape(f.Input, f.Output), g.WithInit(f.init), g.WithName(f.Name))
}

// Fwd is a foward pass on a single fully connected layer.
func (f *FC) Fwd(x *g.Node) (*g.Node, error) {
	prod := g.Must(g.Mul(x, f.weights))
	if f.activation == nil {
		return prod, nil
	}
	a := g.Must(f.activation(prod))
	fmt.Println("a shape: ", a.Shape())
	return a, nil
}

// Learnables are the learnable parameters of the fully connected layer.
func (f *FC) Learnables() g.Nodes {
	return g.Nodes{f.weights}
}
