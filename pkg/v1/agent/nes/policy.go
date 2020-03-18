package nes

import (
	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	modelv1 "github.com/aunum/gold/pkg/v1/model"
	l "github.com/aunum/gold/pkg/v1/model/layers"
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
type LayerBuilder func(x, y *modelv1.Input) []l.Layer

// DefaultFCLayerBuilder is a default fully connected layer builder.
var DefaultFCLayerBuilder = func(x, y *modelv1.Input) []l.Layer {
	return []l.Layer{
		fc.New(x.Squeeze()[0], y.Squeeze()[0], fc.WithActivation(activation.Linear), fc.WithNoBias(), fc.WithName("qvalues")),
	}
}

// MakePolicy makes a model.
func MakePolicy(config *PolicyConfig, base *agentv1.Base, env *envv1.Env) (modelv1.Model, error) {
	x := modelv1.NewInput("state", []int{1, env.ObservationSpaceShape()[0]})
	y := modelv1.NewInput("actionPotentials", []int{1, envv1.PotentialsShape(env.ActionSpace)[0]})

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
