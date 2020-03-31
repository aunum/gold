package nes

import (
	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	"github.com/aunum/goro/pkg/v1/layer"
	modelv1 "github.com/aunum/goro/pkg/v1/model"
)

// PolicyConfig are the hyperparameters for a policy.
type PolicyConfig struct {
	// LayerBuilder is a builder of layer.
	LayerBuilder LayerBuilder

	// Track is whether to track the model.
	Track bool
}

// DefaultPolicyConfig are the default hyperparameters for a policy.
var DefaultPolicyConfig = &PolicyConfig{
	LayerBuilder: DefaultFCLayerBuilder,
	Track:        false,
}

// LayerBuilder builds layers.
type LayerBuilder func(x, y *modelv1.Input) []layer.Config

// DefaultFCLayerBuilder is a default fully connected layer builder.
var DefaultFCLayerBuilder = func(x, y *modelv1.Input) []layer.Config {
	return []layer.Config{
		layer.FC{Input: x.Squeeze()[0], Output: y.Squeeze()[0], Activation: layer.Linear, NoBias: true},
	}
}

// MakePolicy makes a model.
func MakePolicy(config *PolicyConfig, base *agentv1.Base, env *envv1.Env) (modelv1.Model, error) {
	x := modelv1.NewInput("state", env.ObservationSpaceShape())
	x.EnsureBatch()

	y := modelv1.NewInput("actionPotentials", envv1.PotentialsShape(env.ActionSpace))
	y.EnsureBatch()

	base.Logger.Debugv("x shape", x.Shape())
	base.Logger.Debugv("y shape", y.Shape())

	model, err := modelv1.NewSequential("nes")
	if err != nil {
		return nil, err
	}
	model.AddLayers(config.LayerBuilder(x, y)...)

	opts := modelv1.NewOpts()
	if config.Track {
		opts.Add(modelv1.WithTracker(base.Tracker))
	} else {
		opts.Add(modelv1.WithoutTracker())
	}
	opts.Add(modelv1.WithLogger(base.Logger))

	err = model.Compile(x, y, opts.Values()...)
	if err != nil {
		return nil, err
	}
	return model, nil
}
