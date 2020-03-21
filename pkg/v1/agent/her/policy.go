package her

import (
	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	modelv1 "github.com/aunum/gold/pkg/v1/model"
	l "github.com/aunum/gold/pkg/v1/model/layers"

	"github.com/aunum/log"
	g "gorgonia.org/gorgonia"
)

// PolicyConfig are the hyperparameters for a policy.
type PolicyConfig struct {
	// Loss function to evaluate network performance.
	Loss modelv1.Loss

	// Optimizer to optimize the weights with regards to the error.
	Optimizer g.Solver

	// LayerBuilder is a builder of layer.
	LayerBuilder LayerBuilder

	// Batch size to train on.
	BatchSize int

	// Track is whether to track the model.
	Track bool
}

// DefaultPolicyConfig are the default hyperparameters for a policy.
var DefaultPolicyConfig = &PolicyConfig{
	Loss:         modelv1.MSE,
	Optimizer:    g.NewAdamSolver(g.WithBatchSize(128), g.WithLearnRate(0.0005)),
	LayerBuilder: DefaultFCLayerBuilder,
	BatchSize:    128,
	Track:        true,
}

// LayerBuilder builds layers.
type LayerBuilder func(x, y *modelv1.Input) []l.Layer

// DefaultFCLayerBuilder is a default fully connected layer builder.
var DefaultFCLayerBuilder = func(x, y *modelv1.Input) []l.Layer {
	return []l.Layer{
		l.NewFC(x.Squeeze()[0], 512, l.WithActivation(l.ReLU), l.WithName("fc1")),
		l.NewFC(512, 512, l.WithActivation(l.ReLU), l.WithName("fc2")),
		l.NewFC(512, y.Squeeze()[0], l.WithActivation(l.Linear), l.WithName("qvalues")),
	}
}

// MakePolicy makes a model.
func MakePolicy(name string, config *PolicyConfig, base *agentv1.Base, env *envv1.Env) (modelv1.Model, error) {
	stateGoalShape := env.ObservationSpaceShape()[0] * 2
	x := modelv1.NewInput("state_goal", []int{1, stateGoalShape})
	y := modelv1.NewInput("actionPotentials", []int{1, envv1.PotentialsShape(env.ActionSpace)[0]})

	log.Debugv("x shape", x.Shape())
	log.Debugv("y shape", y.Shape())

	model, err := modelv1.NewSequential(name)
	if err != nil {
		return nil, err
	}
	model.AddLayers(config.LayerBuilder(x, y)...)

	opts := modelv1.NewOpts()
	opts.Add(
		modelv1.WithOptimizer(config.Optimizer),
		modelv1.WithLoss(config.Loss),
		modelv1.WithBatchSize(config.BatchSize),
	)
	if config.Track {
		opts.Add(modelv1.WithTracker(base.Tracker))
	} else {
		opts.Add(modelv1.WithoutTracker())
	}

	err = model.Compile(x, y, opts.Values()...)
	if err != nil {
		return nil, err
	}
	return model, nil
}
