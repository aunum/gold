package ppo

import (
	"github.com/aunum/gold/pkg/v1/dense"
	t "gorgonia.org/tensor"
)

// GAE is generalized advantage estimation.
func GAE(values, masks, rewards []*t.Dense, gamma, lambda float32) (returns, advantage *t.Dense, err error) {
	gammaT := t.New(t.WithBacking(gamma))
	lambdaT := t.New(t.WithBacking(lambda))

	gae := t.New(t.WithBacking(float32(0)))
	for i := len(rewards); i >= 0; i-- {
		delta, err := gammaT.Mul(values[i+1])
		if err != nil {
			return nil, nil, err
		}

		// The mask prevents terminal states from being taken into account as they would be applying the
		// start state of the next episode.
		delta, err = delta.Mul(masks[i])
		if err != nil {
			return nil, nil, err
		}
		delta, err = rewards[i].Add(delta)
		if err != nil {
			return nil, nil, err
		}
		delta, err = delta.Sub(values[i])
		if err != nil {
			return nil, nil, err
		}
		gae0, err := gammaT.Mul(lambdaT)
		if err != nil {
			return nil, nil, err
		}
		gae0, err = gae0.Mul(masks[i])
		if err != nil {
			return nil, nil, err
		}
		gae0, err = gae0.Mul(gae)
		if err != nil {
			return nil, nil, err
		}
		gae, err = delta.Add(gae0)
		if err != nil {
			return nil, nil, err
		}
		ret, err := gae.Add(values[i])
		if err != nil {
			return nil, nil, err
		}
		if returns == nil {
			returns = ret
		} else {
			returns, err = ret.Concat(0, returns)
			if err != nil {
				return nil, nil, err
			}
		}
	}
	v, err := dense.Concat(0, values[:len(values)-1]...)
	if err != nil {
		return nil, nil, err
	}
	advantage, err = returns.Sub(v)
	if err != nil {
		return nil, nil, err
	}
	// normalize the advantage.
	advantage, err = dense.ZNorm(advantage)
	if err != nil {
		return nil, nil, err
	}
	return
}
