package layers

import (
	"fmt"

	"gorgonia.org/tensor"

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
	shared     *FC
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

// AsBatch informs the layer compilation that it is a batch.
func AsBatch() func(Layer) {
	return func(l Layer) {
		fc := l.(*FC)
		fc.isBatched = true
	}
}

// Compile the layer into the graph.
// func (f *FC) Compile(x *g.Node, opts ...LayerOpt) {
// 	for _, opt := range opts {
// 		opt(f)
// 	}
// 	if f.shared != nil {
// 		f.weights = g.NewMatrix(x.Graph(), f.dtype, g.WithShape(f.Input, f.Output), g.WithName(f.Name), g.WithValue(f.shared.weights.Value()))
// 		if f.useBias {
// 			f.bias = g.NewMatrix(x.Graph(), f.dtype, g.WithShape(1, f.Output), g.WithName(fmt.Sprintf("%s-bias", f.Name)), g.WithValue(f.shared.bias.Value()))
// 		}
// 		return
// 	}
// 	f.weights = g.NewMatrix(x.Graph(), f.dtype, g.WithShape(f.Input, f.Output), g.WithInit(f.init), g.WithName(f.Name))
// 	if f.useBias {
// 		// Note: the handling of bias here may be wrong.
// 		f.bias = g.NewMatrix(x.Graph(), f.dtype, g.WithShape(1, f.Output), g.WithInit(f.biasInit), g.WithName(fmt.Sprintf("%s-bias", f.Name)))
// 	}
// }

func (f *FC) Compile(x *g.Node, opts ...LayerOpt) {
	f.weights = g.NewMatrix(x.Graph(), f.dtype, g.WithShape(f.Input, f.Output), g.WithInit(f.init), g.WithName(f.Name))
	if f.useBias {
		if f.isBatched {
			f.bias = g.NewMatrix(x.Graph(), f.dtype, g.WithShape(1, f.Output), g.WithInit(f.biasInit))
		} else {
			f.bias = g.NewMatrix(x.Graph(), f.dtype, g.WithShape(x.Shape()[0], f.Output), g.WithInit(f.biasInit))
		}
	}
}

// Fwd is a foward pass on a single fully connected layer.
func (f *FC) Fwd(x *g.Node) (*g.Node, error) {
	var xw, xwb *g.Node
	var err error
	fmt.Println("x shape fwd: ", x.Shape())
	fmt.Println("w shape fwd: ", f.weights.Shape())
	if xw, err = g.Mul(x, f.weights); err != nil {
		return nil, err
	}

	fmt.Println("xw shape: ", xw.Shape())
	if f.bias == nil {
		xwb = xw
		goto act
	}

	fmt.Println("broadcast adding")
	// Note: not sure if this is right to just always broadcast.
	if xwb, err = g.BroadcastAdd(xw, f.bias, nil, []byte{0}); err != nil {
		return nil, err
	}
act:
	if f.activation == nil {
		return xwb, nil
	}
	fmt.Println("xwb: ", xwb.Shape())
	fmt.Printf("activation: %#v\n", f.activation)
	a, err := f.activation.Fwd(xwb)
	if err != nil {
		return nil, err
	}
	fmt.Println("out shape: ", a.Shape())
	return a, err
}

// Learnables are the learnable parameters of the fully connected layer.
func (f *FC) Learnables() g.Nodes {
	if f.bias != nil {
		return g.Nodes{f.weights, f.bias}
	}
	return g.Nodes{f.weights}
}

// Clone the layer without any nodes. (nodes cannot be shared)
func (f *FC) Clone() Layer {
	return &FC{
		Input:      f.Input,
		Output:     f.Output,
		Name:       f.Name,
		activation: f.activation.Clone(),
		init:       f.init,
		dtype:      f.dtype,
		useBias:    f.useBias,
		biasInit:   f.biasInit,
		isBatched:  f.isBatched,
	}
}
