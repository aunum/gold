package deepq

import (
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/pbarker/go-rl/pkg/v1/model"
	"github.com/pbarker/go-rl/pkg/v1/model/layers"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Policy is a dqn policy using a fully connected feed forward neural network.
type Policy struct {
	graph      *g.ExprGraph
	inputs     *g.Node
	prediction *g.Node
	expected   *g.Node
	chain      *model.Chain

	pred    *g.Node
	predVal g.Value
	vm      g.VM
	solver  g.Solver
}

// PolicyConfig is the configuration for a FCPolicy.
type PolicyConfig struct {
	// BatchSize is the size of the batch used to train.
	BatchSize int

	// Type of the network.
	Type tensor.Dtype

	// Cost function to evaluate network perfomance.
	CostFn model.CostFn

	// ChainBuilder is a builder for a chain for layers.
	ChainBuilder ChainBuilder
}

// DefaultPolicyConfig is the default configuration for and FCPolicy.
var DefaultPolicyConfig = &PolicyConfig{
	BatchSize:    100,
	Type:         tensor.Float32,
	CostFn:       model.MeanSquaredError,
	ChainBuilder: DefaultFCChainBuilder,
}

// ChainBuilder is a builder of layer chains for RL puroposes.
type ChainBuilder func(graph *g.ExprGraph, env *envv1.Env) *model.Chain

// DefaultFCChainBuilder creates a default fully connected network for the given action space size.
func DefaultFCChainBuilder(graph *g.ExprGraph, env *envv1.Env) *model.Chain {
	chain := model.NewChain(
		layers.NewFC(g.NewMatrix(graph, g.Float32, g.WithShape(env.ObservationSpaceShape()[0], 2), g.WithName("w0"), g.WithInit(g.GlorotU(1.0))), layers.WithActivation(g.Tanh)),
		layers.NewFC(g.NewMatrix(graph, g.Float32, g.WithShape(2, 100), g.WithName("w1"), g.WithInit(g.GlorotU(1.0))), layers.WithActivation(g.Tanh)),
		layers.NewFC(g.NewMatrix(graph, g.Float32, g.WithShape(100, 100), g.WithName("w2"), g.WithInit(g.GlorotU(1.0))), layers.WithActivation(g.Tanh)),
		layers.NewFC(g.NewMatrix(graph, g.Float32, g.WithShape(100, env.ActionSpaceShape()[0]), g.WithName("w3"), g.WithInit(g.GlorotU(1.0))), layers.WithActivation(g.Tanh)),
		layers.NewFC(g.NewMatrix(graph, g.Float32, g.WithShape(envv1.PotentialsShape(env.ActionSpace)...), g.WithName("w4"), g.WithInit(g.GlorotU(1.0)))),
	)
	return chain
}

// NewPolicy creates a new feed forward policy.
func NewPolicy(c *PolicyConfig, env *envv1.Env) (*Policy, error) {
	graph := g.NewGraph()

	inputs := g.NewMatrix(graph, g.Float32, g.WithShape(c.BatchSize, env.ObservationSpaceShape()[0]), g.WithName("inputs"), g.WithInit(g.Zeroes()))
	expected := g.NewVector(graph, g.Float32, g.WithShape(c.BatchSize), g.WithName("expected"), g.WithInit(g.Zeroes()))

	chain := c.ChainBuilder(graph, env)
	prediction, err := chain.Fwd(inputs)
	if err != nil {
		return nil, err
	}
	cost := c.CostFn(prediction, expected)
	if _, err = g.Grad(cost, chain.Learnables()...); err != nil {
		return nil, err
	}

	vm := g.NewTapeMachine(graph)
	solver := g.NewRMSPropSolver()

	return &Policy{
		graph:      graph,
		inputs:     inputs,
		prediction: prediction,
		expected:   expected,
		chain:      chain,
		vm:         vm,
		solver:     solver,
	}, nil
}

// Predict the correct action given the observation.
func (p *Policy) Predict(observation *tensor.Dense) (qValues *tensor.Dense, err error) {
	err = g.Let(p.inputs, observation)
	if err != nil {
		return qValues, err
	}
	err = p.vm.RunAll()
	if err != nil {
		return qValues, err
	}
	qValues = p.prediction.Value().(*tensor.Dense)
	return
}

// Fit the outcomes.
func (p *Policy) Fit(state, update *tensor.Dense) error {
	g.Let(p.expected, update)
	g.Let(p.inputs, state)
	grads := g.NodesToValueGrads(p.chain.Learnables())

	if err := p.vm.RunAll(); err != nil {
		return err
	}
	p.solver.Step(grads)
	p.vm.Reset()
	return nil
}
