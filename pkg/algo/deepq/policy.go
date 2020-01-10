package deepq

import (
	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Policy for a dqn network.
type Policy interface {
	// Step takes a step given the current observation.
	Step(observation *tensor.Dense) (actions, qValues, states *tensor.Dense, err error)
}

// ActivationFn is an activation funciton for a layer.
type ActivationFn func(x *g.Node) (*g.Node, error)

// FCPolicy is a dqn policy using a fully connected feed forward neural network.
type FCPolicy struct {
	graph   *g.ExprGraph
	inputs  *g.Node
	outputs *g.Node
	layers  []FCLayer

	pred    *g.Node
	predVal g.Value
}

// FCPolicyConfig is the configuration for a FCPolicy.
type FCPolicyConfig struct {
	// BatchSize is the size of the batch used to train.
	BatchSize int

	// Type of the network.
	Type tensor.Dtype
}

// DefaultFCPolicyConfig is the default configuration for and FCPolicy.
var DefaultFCPolicyConfig = &FCPolicyConfig{
	BatchSize: 100,
	Type:      tensor.Float32,
}

// NewFCPolicy creates a new feed forward policy.
func NewFCPolicy(c *FCPolicyConfig, actionSpaceSize int) *FCPolicy {
	graph := g.NewGraph()
	inputs := g.NewMatrix(graph, c.Type, g.WithShape(c.BatchSize, actionSpaceSize), g.WithName("inputs"), g.WithInit(g.Zeroes()))
	outputs := g.NewVector(graph, c.Type, g.WithShape(c.BatchSize), g.WithName("outputs"), g.WithInit(g.Zeroes()))

	layers := []FCLayer{
		FCLayer{Weights: g.NewMatrix(graph, c.Type, g.WithShape(actionSpaceSize, 2), g.WithName("w0"), g.WithInit(g.GlorotU(1.0))), ActivationFn: g.Tanh},
		FCLayer{Weights: g.NewMatrix(graph, c.Type, g.WithShape(2, 100), g.WithName("w1"), g.WithInit(g.GlorotU(1.0))), ActivationFn: g.Tanh},
		FCLayer{Weights: g.NewMatrix(graph, c.Type, g.WithShape(100, 100), g.WithName("w2"), g.WithInit(g.GlorotU(1.0))), ActivationFn: g.Tanh},
		FCLayer{Weights: g.NewMatrix(graph, c.Type, g.WithShape(100, 1), g.WithName("w3"), g.WithInit(g.GlorotU(1.0)))},
	}
	return &FCPolicy{
		graph:   graph,
		inputs:  inputs,
		outputs: outputs,
		layers:  layers,
	}
}

// FCLayer is a fully connected layer of neurons.
type FCLayer struct {
	// Weights for this layer.
	Weights *g.Node

	// ActivationFn is the activation function for this layer.
	ActivationFn ActivationFn
}

// Fwd is a foward pass on a single fully connect layer.
func (l *FCLayer) Fwd(x *g.Node) (*g.Node, error) {
	prod := g.Must(g.Mul(x, l.Weights))
	if l.ActivationFn == nil {
		return prod, nil
	}
	return l.ActivationFn(prod)
}

// Step takes a step given the current observation.
func (p *FCPolicy) Step(observation *tensor.Dense) (actions, qValues, states *tensor.Dense, err error) {

}

func (p *FCPolicy) learnables() g.Nodes {
	retVal := make(g.Nodes, 0, len(p.layers))
	for _, l := range p.layers {
		retVal = append(retVal, l.Weights)
	}
	return retVal
}
