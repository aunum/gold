package ppo

import (
	"github.com/aunum/gold/pkg/v1/common/op"
	modelv1 "github.com/aunum/goro/pkg/v1/model"
	g "gorgonia.org/gorgonia"
)

// Loss is a custom loss for PPO. It is designed to ensure that policies are never
// over updated.
type Loss struct {
	oldProbs, advantages, rewards, values *modelv1.Input
	clippingValue                         float64
	criticDiscount                        float32
	entropyBeta                           float32
}

// LossOpt is an option for PPO loss.
type LossOpt func(*Loss)

// WithClip sets the clipping value.
// Defaults to 0.2
func WithClip(val float64) func(*Loss) {
	return func(l *Loss) {
		l.clippingValue = val
	}
}

// WithCriticDiscount sets the critic discount.
// Defaults to 0.5
func WithCriticDiscount(val float32) func(*Loss) {
	return func(l *Loss) {
		l.criticDiscount = val
	}
}

// WithEntropyBeta sets the entropy beta.
// Defaults to 0.001
func WithEntropyBeta(val float32) func(*Loss) {
	return func(l *Loss) {
		l.entropyBeta = val
	}
}

// NewLoss returns a new PPO loss.
func NewLoss(oldProbs, advantages, rewards, values *modelv1.Input, opts ...LossOpt) *Loss {
	l := &Loss{oldProbs, advantages, rewards, values, 0.2, 0.5, 0.001}
	for _, opt := range opts {
		opt(l)
	}
	return l
}

// Compute the loss.
func (l *Loss) Compute(yHat, y *g.Node) (loss *g.Node, err error) {
	criticDiscount := g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(l.criticDiscount))
	entropyBeta := g.NewScalar(yHat.Graph(), g.Float32, g.WithValue(l.entropyBeta))

	// Find the ratio between the old policy and new policy. Using log for this is computationally cheaper.
	newProbs := g.Must(op.AddFauxF32(yHat))
	newLogProbs := g.Must(g.Log(newProbs))

	oldProbs := g.Must(op.AddFauxF32(l.oldProbs.Node()))
	oldLogProbs := g.Must(g.Log(oldProbs))

	probs := g.Must(g.Sub(newLogProbs, oldLogProbs))
	ratio := g.Must(g.Exp(probs))
	p1 := g.Must(g.BroadcastHadamardProd(ratio, l.advantages.Node(), nil, []byte{1}))

	clipped := g.Must(op.Clip(ratio, 1-l.clippingValue, 1+l.clippingValue))
	p2 := g.Must(g.BroadcastHadamardProd(clipped, l.advantages.Node(), nil, []byte{1}))

	actorLoss := g.Must(op.Min(p1, p2))
	actorLoss = g.Must(g.Mean(actorLoss))
	actorLoss = g.Must(g.Neg(actorLoss))

	criticLoss := g.Must(g.Sub(l.rewards.Node(), l.values.Node()))
	criticLoss = g.Must(g.Square(criticLoss))
	criticLoss = g.Must(g.Mean(criticLoss))

	totalLossProbs := g.Must(g.HadamardProd(yHat, newLogProbs))
	totalLossProbs = g.Must(g.Neg(totalLossProbs))
	totalLossProbs = g.Must(g.Mean(totalLossProbs))

	totalLoss := g.Must(g.HadamardProd(criticDiscount, criticLoss))
	totalLossEnt := g.Must(g.HadamardProd(entropyBeta, totalLossProbs))
	totalLoss = g.Must(g.Add(totalLoss, actorLoss))
	totalLoss = g.Must(g.Sub(totalLoss, totalLossEnt))

	return totalLoss, nil
}

// CloneTo another graph.
func (l *Loss) CloneTo(graph *g.ExprGraph, opts ...modelv1.CloneOpt) modelv1.Loss {
	loss := &Loss{
		oldProbs:       l.oldProbs.CloneTo(graph, opts...),
		advantages:     l.advantages.CloneTo(graph, opts...),
		rewards:        l.advantages.CloneTo(graph, opts...),
		values:         l.advantages.CloneTo(graph, opts...),
		clippingValue:  l.clippingValue,
		criticDiscount: l.criticDiscount,
		entropyBeta:    l.entropyBeta,
	}
	return loss
}

// Inputs returns any inputs the loss function utilizes.
func (l *Loss) Inputs() modelv1.Inputs {
	return modelv1.Inputs{l.advantages, l.oldProbs, l.rewards, l.values}
}
