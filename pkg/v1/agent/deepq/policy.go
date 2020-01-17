package deepq

import (
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/pbarker/go-rl/pkg/v1/model"
	"github.com/pbarker/go-rl/pkg/v1/model/layers"
	"github.com/pbarker/logger"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Policy is a dqn policy using a fully connected feed forward neural network.
type Policy struct {
	graph      *g.ExprGraph
	x          *g.Node
	y          *g.Node
	prediction *g.Node
	chain      *model.Chain

	pred    *g.Node
	predVal g.Value
	vm      g.VM
	solver  g.Solver
	tracker *model.Tracker

	CostNode *g.Node
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
		layers.NewFC(g.NewTensor(graph, g.Float32, 2, g.WithShape(env.ObservationSpaceShape()[0], 24), g.WithName("w0"), g.WithInit(g.GlorotU(1.0))), layers.WithActivation(g.Rectify)),
		layers.NewFC(g.NewTensor(graph, g.Float32, 2, g.WithShape(24, 24), g.WithName("w1"), g.WithInit(g.GlorotU(1.0))), layers.WithActivation(g.Rectify)),
		layers.NewFC(g.NewTensor(graph, g.Float32, 2, g.WithShape(24, envv1.PotentialsShape(env.ActionSpace)[0]), g.WithName("w2"), g.WithInit(g.GlorotU(1.0)))),
	)
	return chain
}

// NewPolicy creates a new policy.
func NewPolicy(c *PolicyConfig, env *envv1.Env) (*Policy, error) {
	graph := g.NewGraph()
	tracker := model.NewTracker(graph)

	x := g.NewTensor(graph, g.Float32, 1, g.WithShape(env.ObservationSpaceShape()[0]), g.WithName("x"), g.WithInit(g.Zeroes()))
	y := g.NewTensor(graph, g.Float32, 1, g.WithShape(envv1.PotentialsShape(env.ActionSpace)[0]), g.WithName("y"), g.WithInit(g.Zeroes()))

	chain := c.ChainBuilder(graph, env)
	prediction, err := chain.Fwd(x)
	if err != nil {
		return nil, err
	}
	// prediction, err = g.SoftMax(prediction)
	// if err != nil {
	// 	return nil, err
	// }

	cost := c.CostFn(prediction, y)

	_, err = g.Grad(cost, chain.Learnables()...)
	if err != nil {
		return nil, err
	}

	vm := g.NewTapeMachine(graph, g.BindDualValues(chain.Learnables()...))
	solver := g.NewAdamSolver()

	p := &Policy{
		graph:      graph,
		x:          x,
		y:          y,
		prediction: prediction,
		chain:      chain,
		vm:         vm,
		solver:     solver,
		tracker:    tracker,
		CostNode:   cost,
	}
	return p, nil
}

// Predict the correct action given the input.
func (p *Policy) Predict(x *tensor.Dense) (qValues *tensor.Dense, err error) {
	logger.Info("x in: ", x)
	err = g.Let(p.x, x)
	if err != nil {
		return qValues, err
	}
	logger.Info("node length: ", p.graph.Nodes().Len())
	err = p.vm.RunAll()
	if err != nil {
		return qValues, err
	}
	p.vm.Reset()
	qValues = p.prediction.Value().(*tensor.Dense)
	return
}

// Fit x to y.
func (p *Policy) Fit(x, y *tensor.Dense) error {
	g.Let(p.y, y)
	g.Let(p.x, x)
	grads := g.NodesToValueGrads(p.chain.Learnables())
	logger.Info("cost pre fit: ", p.CostNode.Value())
	err := p.vm.RunAll()
	if err != nil {
		return err
	}
	qValues := p.prediction.Value().(*tensor.Dense)
	logger.Info("qvalues: ", qValues)
	logger.Info("cost during fit: ", p.CostNode.Value())
	p.solver.Step(grads)
	p.vm.Reset()
	return nil
}
