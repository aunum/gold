package reinforce

import (
	agentv1 "github.com/pbarker/go-rl/pkg/v1/agent"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	modelv1 "github.com/pbarker/go-rl/pkg/v1/model"
	l "github.com/pbarker/go-rl/pkg/v1/model/layers"

	"github.com/pbarker/log"
	g "gorgonia.org/gorgonia"
)

// PolicyConfig are the hyperparameters for a policy.
type PolicyConfig struct {
	// Optimizer to optimize the wieghts with regards to the error.
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
type LayerBuilder func(x, y *modelv1.Input) []l.Layer

// DefaultFCLayerBuilder is a default fully connected layer builder.
var DefaultFCLayerBuilder = func(x, y *modelv1.Input) []l.Layer {
	return []l.Layer{
		l.NewFC(x.Squeeze()[0], 24, l.WithActivation(l.ReLU), l.WithName("fc1")),
		l.NewFC(24, 24, l.WithActivation(l.ReLU), l.WithName("fc2")),
		l.NewFC(24, y.Squeeze()[0], l.WithActivation(l.NewSoftmax()), l.WithName("dist")),
	}
}

// MakePolicy makes a model.
func MakePolicy(config *PolicyConfig, base *agentv1.Base, env *envv1.Env) (modelv1.Model, error) {
	x := modelv1.NewInput("state", []int{1, env.ObservationSpaceShape()[0]})
	y := modelv1.NewInput("advantages", []int{1, envv1.PotentialsShape(env.ActionSpace)[0]})

	log.Infov("x shape", x.Shape())
	log.Infov("y shape", y.Shape())

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
