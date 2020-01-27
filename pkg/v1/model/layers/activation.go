package layers

import (
	"fmt"

	g "gorgonia.org/gorgonia"
)

// Activation is an activation function.
type Activation interface {
	// Fwd is a foward pass through x.
	Fwd(x *g.Node) (*g.Node, error)

	// Clone the activation.
	Clone() Activation
}

// SigmoidActivation is a sigmoid activation layer.
type SigmoidActivation struct{}

// Sigmoid returns a new sigmoid activation layer.
func Sigmoid() *SigmoidActivation {
	return &SigmoidActivation{}
}

// Fwd is a foward pass through the layer.
func (s *SigmoidActivation) Fwd(x *g.Node) (*g.Node, error) {
	return g.Sigmoid(x)
}

// Learnables returns all learnable nodes within this layer.
func (s *SigmoidActivation) Learnables() (n g.Nodes) {
	return n
}

// Compile the layer.
func (s *SigmoidActivation) Compile(x *g.Node, opts ...LayerOpt) {}

// Clone the activation.
func (s *SigmoidActivation) Clone() Activation {
	return Sigmoid()
}

// TanhActivation is a tanh activation layer.
type TanhActivation struct{}

// Tanh returns a new tanh activation layer.
func Tanh() *TanhActivation {
	return &TanhActivation{}
}

// Fwd is a foward pass through the layer.
func (t *TanhActivation) Fwd(x *g.Node) (*g.Node, error) {
	return g.Tanh(x)
}

// Learnables returns all learnable nodes within this layer.
func (t *TanhActivation) Learnables() (n g.Nodes) {
	return n
}

// Compile the layer.
func (t *TanhActivation) Compile(x *g.Node, opts ...LayerOpt) {}

// Clone the activation.
func (t *TanhActivation) Clone() Activation {
	return Tanh()
}

// ReLUActivation is a relu activation layer.
type ReLUActivation struct{}

// ReLU returns a new relu activation layer.
func ReLU() *ReLUActivation {
	return &ReLUActivation{}
}

// Fwd is a foward pass through the layer.
func (r *ReLUActivation) Fwd(x *g.Node) (*g.Node, error) {
	return g.Rectify(x)
}

// Learnables returns all learnable nodes within this layer.
func (r *ReLUActivation) Learnables() (n g.Nodes) {
	return n
}

// Compile the layer.
func (r *ReLUActivation) Compile(x *g.Node, opts ...LayerOpt) {}

// Clone the activation.
func (r *ReLUActivation) Clone() Activation {
	return ReLU()
}

// LeakyReLUActivation is a leaky relu activation layer.
type LeakyReLUActivation struct {
	alpha float64
}

// LeakyReLU returns a new leaky relu activation layer.
func LeakyReLU(alpha float64) *LeakyReLUActivation {
	return &LeakyReLUActivation{alpha: alpha}
}

// Fwd is a foward pass through the layer.
func (r *LeakyReLUActivation) Fwd(x *g.Node) (*g.Node, error) {
	return g.LeakyRelu(x, r.alpha)
}

// Learnables returns all learnable nodes within this layer.
func (r *LeakyReLUActivation) Learnables() (n g.Nodes) {
	return n
}

// Compile the layer.
func (r *LeakyReLUActivation) Compile(x *g.Node, opts ...LayerOpt) {}

// Clone the activation.
func (r *LeakyReLUActivation) Clone() Activation {
	return LeakyReLU(r.alpha)
}

// SoftmaxActivation is a softmax activation layer.
type SoftmaxActivation struct {
	axis []int
}

// Softmax returns a new leaky softmax activation layer.
func Softmax(axis ...int) *SoftmaxActivation {
	// if len(axis) == 0 {
	// 	axis = append(axis, 0)
	// }
	return &SoftmaxActivation{axis: axis}
}

// Fwd is a foward pass through the layer.
func (s *SoftmaxActivation) Fwd(x *g.Node) (*g.Node, error) {
	fmt.Printf("running softmax with x shape: %v dims: %v \n", x.Shape(), x.Dims())
	return g.SoftMax(x, s.axis...)
}

// Learnables returns all learnable nodes within this layer.
func (s *SoftmaxActivation) Learnables() (n g.Nodes) {
	return n
}

// Compile the layer.
func (s *SoftmaxActivation) Compile(x *g.Node, opts ...LayerOpt) {}

// Clone the activation.
func (s *SoftmaxActivation) Clone() Activation {
	return Softmax(s.axis...)
}

// LinearActivation is a linear (identity) activation layer.
type LinearActivation struct{}

// Linear is a linear activation layer.
func Linear() *LinearActivation {
	return &LinearActivation{}
}

// Fwd is a foward pass through the layer.
func (l *LinearActivation) Fwd(x *g.Node) (*g.Node, error) {
	return x, nil
}

// Learnables returns all learnable nodes within this layer.
func (l *LinearActivation) Learnables() (n g.Nodes) {
	return n
}

// Compile the layer.
func (l *LinearActivation) Compile(x *g.Node, opts ...LayerOpt) {}

// Clone the activation.
func (l *LinearActivation) Clone() Activation {
	return Linear()
}
