package ppo1

import (
	"fmt"

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
	newProbs := g.Must(g.Add(yHat, g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(float32(1e-10)))))
	newLogProbs := g.Must(g.Log(newProbs))

	oldProbs := g.Must(g.Add(p.oldProbs.Node(), g.NewScalar(p.oldProbs.Node().Graph(), g.Float32, g.WithValue(float32(1e-10)))))
	oldLogProbs := g.Must(g.Log(oldProbs))

	probs := g.Must(g.Sub(newLogProbs, oldLogProbs))
	ratio := g.Must(g.Exp(probs))
	p1 := g.Must(g.Mul(ratio, p.advantages.Node()))

	clipped := g.Must(op.Clip(ratio, 1-p.clippingValue, 1+p.clippingValue))
	p2 := g.Must(g.Mul(clipped, p.advantages.Node()))

	actorLoss := g.Must(op.Min(p1, p2))
	actorLoss = g.Must(g.Mean(actorLoss))
	actorLoss = g.Must(g.Neg(actorLoss))

	criticLoss := g.Must(g.Sub(p.rewards.Node(), p.values.Node()))
	criticLoss = g.Must(g.Square(criticLoss))
	criticLoss = g.Must(g.Mean(criticLoss))

	totalLossProbs := g.Must(g.Mul(yHat, newLogProbs))
	totalLossProbs = g.Must(g.Neg(totalLossProbs))
	totalLossProbs = g.Must(g.Mean(totalLossProbs))

	totalLoss := g.Must(g.Mul(criticDiscount, criticLoss))
	totalLossEnt := g.Must(g.Mul(entropyBeta, totalLossProbs))
	totalLoss = g.Must(g.Add(totalLoss, actorLoss))
	totalLoss = g.Must(g.Sub(totalLoss, totalLossEnt))
	fmt.Println("hello")
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
