package deepq

import (
	agentv1 "github.com/pbarker/go-rl/pkg/v1/agent"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	modelv1 "github.com/pbarker/go-rl/pkg/v1/model"
	l "github.com/pbarker/go-rl/pkg/v1/model/layers"
	"github.com/pbarker/log"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ModelConfig are the hyperparameters for a model.
type ModelConfig struct {
	// Loss function to evaluate network perfomance.
	LossFn modelv1.LossFn

	// Optimizer to optimize the wieghts with regards to the error.
	Optimizer g.Solver

	// LayerBuilder is a builder of layer.
	LayerBuilder LayerBuilder

	// BatchSize of the updates.
	BatchSize int

	// Track is whether to track the model.
	Track bool
}

// DefaultActorConfig are the default hyperparameters for a policy.
var DefaultActorConfig = &ModelConfig{
	LossFn:       modelv1.MeanSquaredError,
	Optimizer:    g.NewAdamSolver(g.WithLearnRate(0.001)),
	LayerBuilder: DefaultActorLayerBuilder,
	BatchSize:    20,
	Track:        true,
}

// LayerBuilder builds layers.
type LayerBuilder func(env *envv1.Env) []l.Layer

// DefaultActorLayerBuilder is a default fully connected layer builder.
var DefaultActorLayerBuilder = func(env *envv1.Env) []l.Layer {
	return []l.Layer{
		l.NewFC(env.ObservationSpaceShape()[0], 24, l.WithActivation(l.ReLU), l.WithName("fc1")),
		l.NewFC(24, 24, l.WithActivation(l.ReLU), l.WithName("fc2")),
		l.NewFC(24, envv1.PotentialsShape(env.ActionSpace)[0], l.WithActivation(l.Softmax), l.WithName("predictions")),
	}
}

// MakeActor makes the actor which chooses actions based on the policy.
func MakeActor(config *ModelConfig, base *agentv1.Base, env *envv1.Env) (modelv1.Model, error) {
	x := modelv1.NewInput("x", []int{1, env.ObservationSpaceShape()[0]})
	y := modelv1.NewInput("y", []int{1, envv1.PotentialsShape(env.ActionSpace)[0]})

	oldPolicyProbabilities := modelv1.NewInput("oldProbs", []int{1, envv1.PotentialsShape(env.ActionSpace)[0]})
	advantages := modelv1.NewInput("advantages", []int{1, 1})
	rewards := modelv1.NewInput("rewards", []int{1, 1})
	values := modelv1.NewInput("values", []int{1, 1})

	log.Infov("xshape", x.Shape())
	log.Infov("yshape", y.Shape())

	model, err := modelv1.NewSequential("actor")
	if err != nil {
		return nil, err
	}
	model.AddLayers(config.LayerBuilder(env)...)

	opts := modelv1.NewOpts()
	opts.Add(
		modelv1.WithOptimizer(config.Optimizer),
		modelv1.WithLoss(config.LossFn),
		modelv1.WithBatchSize(config.BatchSize),
	)
	if config.Track {
		opts.Add(modelv1.WithTracker(base.Tracker))
	} else {
		opts.Add(modelv1.WithNoTracker())
	}
	model.Fwd(x)
	err = model.Compile(
		modelv1.Inputs{x, oldPolicyProbabilities, advantages, rewards, values},
		y,
		opts.Values()...,
	)
	if err != nil {
		return nil, err
	}
	return nil, nil
}

// DefaultCriticLayerBuilder is a default fully connected layer builder.
var DefaultCriticLayerBuilder = func(env *envv1.Env) []l.Layer {
	return []l.Layer{
		l.NewFC(env.ObservationSpaceShape()[0], 24, l.WithActivation(l.ReLU()), l.WithName("w0")),
		l.NewFC(24, 24, l.WithActivation(l.ReLU()), l.WithName("w1")),
		l.NewFC(24, 1, l.WithActivation(l.Tanh()), l.WithName("w2")),
	}
}

// MakeCritic makes the critic which creats a qValue based on the outcome of the action taken.
func MakeCritic(config *ModelConfig) (modelv1.Model, error) {
	return nil, nil
}

// GAE is generalized advantage estimation.
func GAE(values, masks, rewards []float32) float32 {
	return 0
}

/*
def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
	return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
*/

// PPOLoss is a custom loss for PPO. It is designed to ensure that policies are never
// updated more than the previous.
type PPOLoss struct {
	oldProbs, advantages, rewards, values *modelv1.Input
}

// NewPPOLoss returns a new PPO loss.
func NewPPOLoss(oldProbs, advantages, rewards, values *modelv1.Input) *PPOLoss {
	return &PPOLoss{oldProbs, advantages, rewards, values}
}

// Compute the loss
func (p *PPOLoss) Compute(yHat, y *g.Node) (loss *g.Node, err error) {
	newProbs := yHat

	// Find the ratio between the old policy and new policy. Using log for this is computationally cheaper.
	ratio, err := g.Exp(g.Must(g.Log(newProbs+1e-10)) - g.Must(g.Log(p.oldProbs.Node()+1e-10)))
	if err != nil {
		return err
	}
	p1, err := g.Mul(ratio, p.advantages)
	if err != nil {
		return err
	}
	p2 := 
	

	return
}

// CloneTo another graph.
func (p *PPOLoss) CloneTo(graph *g.ExprGraph) modelv1.Loss {
	return p
}

/*
def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

	return loss
*/

// MakePolicy makes a model.
func MakePolicy(name string, config *ModelConfig, base *agentv1.Base, env *envv1.Env) (modelv1.Model, error) {
	x := tensor.Ones(tensor.Float32, 1, env.ObservationSpaceShape()[0])
	y := tensor.Ones(tensor.Float32, 1, envv1.PotentialsShape(env.ActionSpace)[0])

	log.Infov("xshape", x.Shape())
	log.Infov("yshape", y.Shape())

	model, err := modelv1.NewSequential(name)
	if err != nil {
		return nil, err
	}
	model.AddLayers(config.LayerBuilder(env)...)

	opts := modelv1.NewOpts()
	opts.Add(
		modelv1.WithOptimizer(config.Optimizer),
		modelv1.WithLoss(config.LossFn),
		modelv1.WithBatchSize(config.BatchSize),
	)
	if config.Track {
		opts.Add(modelv1.WithTracker(base.Tracker))
	} else {
		opts.Add(modelv1.WithNoTracker())
	}

	err = model.Compile(x, y, opts.Values()...)
	if err != nil {
		return nil, err
	}
	return model, nil
}
