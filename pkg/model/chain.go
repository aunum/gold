package model

import (
	"gorgonia.org/tensor"
	g "gorgonia.org/gornonia"
)

// Chain of layers.
type Chain struct {
	// Layers are the layers to chain together.
	Layers []Layer
}

// NewChain returns a new chain of layers.
func NewChain(layers ...Layer) *Chain {
	return &Chain{
		Layers: layers,
	}
}

// Fwd is a forward pass thorugh all layers of the chain.
func (c *Chain) Fwd(inputs *g.Node) (pred *g.Node) {
	pred := inputs
	for _, layer := range c.Layers {
		if pred, err = layer.fwd(pred); err != nil {
			return nil, err
		}
	}
	return pred
}

// Learnables are all of the learnable parameters in the chain.
func (c *Chain) Learnables() g.Nodes {
	retVal := make(g.Nodes, 0, len(c.Layers))
	for _, l := range p.layers {
		retVal = append(retVal, l.Learnables()...)
	}
	return retVal
}