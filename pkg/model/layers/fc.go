package layers

import (
	"gorgonia.org/tensor"
	g "gorgonia.org/gornonia"
	"github.com/pbarker/go-rl/pkg/model"
)

// FC is a fully connected layer of neurons.
type FC struct {
	// Weights for this layer.
	Weights *g.Node

	// activationFn is the activation function for this layer.
	activationFn model.ActivationFn
}

// NewFC returns a new fully connected layer.
func NewFC(weights *g.Node, opts ...FCOpts) *FC {
	return &FC{
		Weights: wieghts,
		
	}
}

// Fwd is a foward pass on a single fully connect layer.
func (l *FCLayer) Fwd(x *g.Node) (*g.Node, error) {
	prod := g.Must(g.Mul(x, l.Weights))
	if l.ActivationFn == nil {
		return prod, nil
	}
	return l.ActivationFn(prod)
}

// Learnables are the learnable parameters of the fully connected layer.
func (l *FCLayer) Learnables() g.Nodes {
	return l.Weights
}