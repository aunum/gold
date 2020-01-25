package deepq

import (
	agentv1 "github.com/pbarker/go-rl/pkg/v1/agent"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	modelv1 "github.com/pbarker/go-rl/pkg/v1/model"
	l "github.com/pbarker/go-rl/pkg/v1/model/layers"
	"github.com/pbarker/logger"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// PolicyConfig are the hyperparameters for a policy.
type PolicyConfig struct {
	// Loss function to evaluate network perfomance.
	LossFn modelv1.LossFn

	// Optimizer to optimize the wieghts with regards to the error.
	Optimizer g.Solver

	// LayerBuilder is a builder of layer.
	LayerBuilder LayerBuilder
}

// DefaultPolicyConfig are the default hyperparameters for a policy.
var DefaultPolicyConfig = &PolicyConfig{
	LossFn:       modelv1.MeanSquaredError,
	Optimizer:    g.NewAdamSolver(g.WithLearnRate(0.001)),
	LayerBuilder: DefaultFCLayerBuilder,
}

// LayerBuilder builds layers.
type LayerBuilder func(env *envv1.Env) []modelv1.Layer

// DefaultFCLayerBuilder is a default fully connected layer builder.
var DefaultFCLayerBuilder = func(env *envv1.Env) []modelv1.Layer {
	return []modelv1.Layer{
		l.NewFC(env.ObservationSpaceShape()[0], 24, l.WithActivation(l.ReLU()), l.WithName("w0")),
		l.NewFC(24, 24, l.WithActivation(l.ReLU()), l.WithName("w1")),
		l.NewFC(24, envv1.PotentialsShape(env.ActionSpace)[0], l.WithActivation(l.Linear()), l.WithName("w2")),
	}
}

// MakePolicy makes a model.
func MakePolicy(name string, config *PolicyConfig, base *agentv1.Base, env *envv1.Env) (modelv1.Model, error) {
	x := tensor.Ones(tensor.Float32, 1, env.ObservationSpaceShape()[0])
	y := tensor.Ones(tensor.Float32, 1, envv1.PotentialsShape(env.ActionSpace)[0])

	logger.Infov("xshape", x.Shape())
	logger.Infov("yshape", y.Shape())

	model, err := modelv1.NewSequential(name, x, y)
	if err != nil {
		return nil, err
	}
	model.AddLayers(config.LayerBuilder(env)...)

	err = model.Compile(
		modelv1.WithOptimizer(config.Optimizer),
		modelv1.WithLoss(config.LossFn),
		modelv1.WithTracker(base.Tracker),
	)
	if err != nil {
		return nil, err
	}
	return model, nil
}
