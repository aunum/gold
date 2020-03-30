package deepq

import (
	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	modelv1 "github.com/aunum/goro/pkg/v1/model"
	"github.com/aunum/goro/pkg/v1/layer"

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

	// BatchSize of the updates.
	BatchSize int

	// Track is whether to track the model.
	Track bool
}

// DefaultPolicyConfig are the default hyperparameters for a policy.
var DefaultPolicyConfig = &PolicyConfig{
	Loss:         modelv1.MSE,
	Optimizer:    g.NewAdamSolver(g.WithLearnRate(0.0005)),
	LayerBuilder: DefaultFCLayerBuilder,
	BatchSize:    20,
	Track:        true,
}

// LayerBuilder builds layers.
type LayerBuilder func(x, y *modelv1.Input) []layer.Config

// DefaultFCLayerBuilder is a default fully connected layer builder.
var DefaultFCLayerBuilder = func(x, y *modelv1.Input) []layer.Config {
	return []layer.Config{
		layer.FC{Input: x.Squeeze()[0], Output: 24},
		layer.FC{Input: 24, Output: 24},
		layer.FC{Input: 24, Output: y.Squeeze()[0], Activation: layer.Linear},
	}
}

// DefaultAtariLayerBuilder is a default fully connected layer builder.
var DefaultAtariLayerBuilder = func(x, y *modelv1.Input) []layer.Config {
	return []layer.Config{
		layer.Conv2D{Input: 1, Output: 32, Width: 3, Height: 3},
		layer.Conv2D{Input: 32, Output: 64, Width: 3, Height: 3},
		layer.Conv2D{Input: 64, Output: 64, Width: 3, Height: 3},
		layer.Flatten{},
		layer.FC{Input: 12800, Output: 24},
		layer.FC{Input: 24, Output: y.Squeeze()[0], Activation: layer.Linear},
	}
}

// MakePolicy makes a policy model.
func MakePolicy(name string, config *PolicyConfig, base *agentv1.Base, env *envv1.Env) (modelv1.Model, error) {
	x := modelv1.NewInput("state", env.ObservationSpaceShape())
	x.EnsureBatch()

	y := modelv1.NewInput("actionPotentials", envv1.PotentialsShape(env.ActionSpace))
	y.EnsureBatch()

	log.Infov("x shape", x.Shape())
	log.Infov("y shape", y.Shape())

	model, err := modelv1.NewSequential(name)
	if err != nil {
		panic(err)
		// return nil, err
	}
	model.AddLayers(config.LayerBuilder(x, y)...)

	opts := modelv1.NewOpts()
	opts.Add(
		modelv1.WithOptimizer(config.Optimizer),
		modelv1.WithLoss(config.Loss),
		modelv1.WithBatchSize(config.BatchSize),
		modelv1.WithMetrics(modelv1.TrainBatchLossMetric),
	)
	if config.Track {
		opts.Add(modelv1.WithTracker(base.Tracker))
	} else {
		opts.Add(modelv1.WithoutTracker())
	}

	err = model.Compile(x, y, opts.Values()...)
	if err != nil {
		panic(err)
		// return nil, err
	}
	return model, nil
}
