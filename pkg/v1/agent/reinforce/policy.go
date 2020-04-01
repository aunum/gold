package reinforce

import (
	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	"github.com/aunum/goro/pkg/v1/layer"
	modelv1 "github.com/aunum/goro/pkg/v1/model"

	"github.com/aunum/log"
	g "gorgonia.org/gorgonia"
)

// PolicyConfig are the hyperparameters for a policy.
type PolicyConfig struct {
	// Optimizer to optimize the weights with regards to the error.
	Optimizer g.Solver

	// LayerBuilder is a builder of layer.
	LayerBuilder LayerBuilder

	// Track is whether to track the model.
	Track bool
}

// DefaultPolicyConfig are the default hyperparameters for a policy.
var DefaultPolicyConfig = &PolicyConfig{
	Optimizer:    g.NewAdamSolver(),
	LayerBuilder: DefaultFCLayerBuilder,
	Track:        true,
}

// LayerBuilder builds layers.
type LayerBuilder func(x, y *modelv1.Input) []layer.Config

// DefaultFCLayerBuilder is a default fully connected layer builder.
var DefaultFCLayerBuilder = func(x, y *modelv1.Input) []layer.Config {
	return []layer.Config{
		layer.FC{Input: x.Squeeze()[0], Output: 24},
		layer.FC{Input: 24, Output: 24},
		layer.FC{Input: 24, Output: y.Squeeze()[0], Activation: layer.Softmax},
	}
}

// MakePolicy makes a model.
func MakePolicy(config *PolicyConfig, base *agentv1.Base, env *envv1.Env) (modelv1.Model, error) {
	x := modelv1.NewInput("state", env.ObservationSpaceShape())
	x.EnsureBatch()

	y := modelv1.NewInput("advantages", envv1.PotentialsShape(env.ActionSpace))
	y.EnsureBatch()

	log.Debugv("x shape", x.Shape())
	log.Debugv("y shape", y.Shape())

	model, err := modelv1.NewSequential("reinforce")
	if err != nil {
		return nil, err
	}
	model.AddLayers(config.LayerBuilder(x, y)...)

	opts := modelv1.NewOpts()
	opts.Add(
		modelv1.WithOptimizer(config.Optimizer),
		modelv1.WithLoss(modelv1.CrossEntropy),
		modelv1.WithMetrics(modelv1.TrainBatchLossMetric),
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
