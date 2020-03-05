package reinforce

import (
	modelv1 "github.com/pbarker/go-rl/pkg/v1/model"
	g "gorgonia.org/gorgonia"
)

// Loss is a custom loss for PPO. It is designed to ensure that policies are never
// over updated.
type Loss struct {
	discountedRewards *modelv1.Input
}

// LossOpt is an option for loss.
type LossOpt func(*Loss)

// NewLoss returns a new PPO loss.
func NewLoss(discountedRewards *modelv1.Input) *Loss {
	return &Loss{discountedRewards}
}

// Compute the loss.
func (l *Loss) Compute(yHat, y *g.Node) (loss *g.Node, err error) {
	loss, err = modelv1.CrossEntropy.Compute(yHat, y)
	if err != nil {
		return
	}
	loss, err = g.BroadcastHadamardProd(loss, l.discountedRewards.Node(), []byte{}, []byte{})
	return
}

// CloneTo another graph.
func (l *Loss) CloneTo(graph *g.ExprGraph, opts ...modelv1.CloneOpt) modelv1.Loss {
	return &Loss{
		discountedRewards: l.discountedRewards.CloneTo(graph, opts...),
	}
}

// Inputs returns any inputs the loss function utilizes.
func (l *Loss) Inputs() modelv1.Inputs {
	return modelv1.Inputs{l.discountedRewards}
}
