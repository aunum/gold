package ppo1

import (
	"github.com/pbarker/go-rl/pkg/v1/common/op"
	modelv1 "github.com/pbarker/go-rl/pkg/v1/model"
	g "gorgonia.org/gorgonia"
)

// PPOLoss is a custom loss for PPO. It is designed to ensure that policies are never
// over updated.
type PPOLoss struct {
	oldProbs, advantages, rewards, values *modelv1.Input
	clippingValue                         float64
	criticDiscount                        float32
	entropyBeta                           float32
}

// LossOpt is an option for PPO loss.
type LossOpt func(*PPOLoss)

// WithClip sets the clipping value.
// Defaults to 0.2
func WithClip(val float64) func(*PPOLoss) {
	return func(p *PPOLoss) {
		p.clippingValue = val
	}
}

// WithCriticDiscount sets the critic discount.
// Defaults to 0.5
func WithCriticDiscount(val float32) func(*PPOLoss) {
	return func(p *PPOLoss) {
		p.criticDiscount = val
	}
}

// WithEntropyBeta sets the entropy beta.
// Defaults to 0.001
func WithEntropyBeta(val float32) func(*PPOLoss) {
	return func(p *PPOLoss) {
		p.entropyBeta = val
	}
}

// NewPPOLoss returns a new PPO loss.
func NewPPOLoss(oldProbs, advantages, rewards, values *modelv1.Input, opts ...LossOpt) *PPOLoss {
	l := &PPOLoss{oldProbs, advantages, rewards, values, 0.2, 0.5, 0.001}
	for _, opt := range opts {
		opt(l)
	}
	return l
}

// Compute the loss.
func (p *PPOLoss) Compute(yHat, y *g.Node) (loss *g.Node, err error) {
	criticDiscount := g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(p.criticDiscount))
	entropyBeta := g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(p.entropyBeta))

	// Find the ratio between the old policy and new policy. Using log for this is computationally cheaper.
	newProbs, err := g.Add(yHat, g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(1e-10)))
	if err != nil {
		return nil, err
	}
	newLogProbs, err := g.Log(newProbs)
	if err != nil {
		return nil, err
	}

	oldProbs, err := g.Add(p.oldProbs.Node(), g.NewScalar(p.oldProbs.Node().Graph(), g.Float32, g.WithValue(1e-10)))
	if err != nil {
		return nil, err
	}
	oldLogProbs, err := g.Log(oldProbs)
	if err != nil {
		return nil, err
	}

	probs, err := g.Sub(newLogProbs, oldLogProbs)
	if err != nil {
		return nil, err
	}

	ratio, err := g.Exp(probs)
	if err != nil {
		return nil, err
	}
	p1, err := g.Mul(ratio, p.advantages.Node())
	if err != nil {
		return nil, err
	}

	clipped, err := op.Clip(ratio, 1-p.clippingValue, 1+p.clippingValue)
	if err != nil {
		return nil, err
	}
	p2, err := g.Mul(clipped, p.advantages.Node())
	if err != nil {
		return nil, err
	}

	actorLoss, err := op.Min(p1, p2)
	if err != nil {
		return nil, err
	}
	actorLoss, err = g.Mean(actorLoss)
	if err != nil {
		return nil, err
	}
	actorLoss, err = g.Neg(actorLoss)
	if err != nil {
		return nil, err
	}

	criticLoss, err := g.Sub(p.rewards.Node(), p.values.Node())
	if err != nil {
		return nil, err
	}
	criticLoss, err = g.Square(criticLoss)
	if err != nil {
		return nil, err
	}
	criticLoss, err = g.Mean(criticLoss)
	if err != nil {
		return nil, err
	}

	totalLossProbs, err := g.Mul(yHat, newLogProbs)
	if err != nil {
		return nil, err
	}
	totalLossProbs, err = g.Neg(totalLossProbs)
	if err != nil {
		return nil, err
	}
	totalLossProbs, err = g.Mean(totalLossProbs)
	if err != nil {
		return nil, err
	}

	totalLoss, err := g.Mul(criticDiscount, criticLoss)
	if err != nil {
		return nil, err
	}
	totalLossEnt, err := g.Mul(entropyBeta, totalLossProbs)
	if err != nil {
		return nil, err
	}
	totalLoss, err = g.Add(totalLoss, actorLoss)
	if err != nil {
		return nil, err
	}
	totalLoss, err = g.Sub(totalLoss, totalLossEnt)
	if err != nil {
		return nil, err
	}

	return totalLoss, nil
}

// CloneTo another graph.
func (p *PPOLoss) CloneTo(graph *g.ExprGraph) modelv1.Loss {
	l := &PPOLoss{
		oldProbs:       p.oldProbs.CloneTo(graph),
		advantages:     p.advantages.CloneTo(graph),
		rewards:        p.advantages.CloneTo(graph),
		values:         p.advantages.CloneTo(graph),
		clippingValue:  p.clippingValue,
		criticDiscount: p.criticDiscount,
		entropyBeta:    p.entropyBeta,
	}
	return l
}
