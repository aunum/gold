package layers

import (
	"github.com/pbarker/go-rl/pkg/v1/model"
	g "gorgonia.org/gorgonia"
)

// Activation is an activation function.
type Activation interface {
	// Fwd is a foward pass through x.
	Fwd(x *g.Node) (*g.Node, error)
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
func (s *SigmoidActivation) Compile(model model.Model) {}

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
func (t *TanhActivation) Compile(model model.Model) {}

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
func (r *ReLUActivation) Compile(model model.Model) {}

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
func (r *LeakyReLUActivation) Compile(model model.Model) {}

// SoftmaxActivation is a softmax activation layer.
type SoftmaxActivation struct {
	axis []int
}

// Softmax returns a new leaky relu activation layer.
func Softmax(axis ...int) *SoftmaxActivation {
	return &SoftmaxActivation{axis: axis}
}

// Fwd is a foward pass through the layer.
func (s *SoftmaxActivation) Fwd(x *g.Node) (*g.Node, error) {
	return g.SoftMax(x, s.axis...)
}

// Learnables returns all learnable nodes within this layer.
func (s *SoftmaxActivation) Learnables() (n g.Nodes) {
	return n
}

// Compile the layer.
func (s *SoftmaxActivation) Compile(model model.Model) {}

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
func (l *LinearActivation) Compile(model model.Model) {}
