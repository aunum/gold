package model

import (
	g "gorgonia.org/gorgonia"
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
func (c *Chain) Fwd(inputs *g.Node) (prediction *g.Node, err error) {
	prediction = inputs
	for _, layer := range c.Layers {
		if prediction, err = layer.Fwd(prediction); err != nil {
			return nil, err
		}
	}
	return prediction, nil
}

// Learnables are all of the learnable parameters in the chain.
func (c *Chain) Learnables() g.Nodes {
	retVal := []*g.Node{}
	for _, layer := range c.Layers {
		retVal = append(retVal, layer.Learnables()...)
	}
	return retVal
}

// Add to the chain.
func (c *Chain) Add(l ...Layer) {
	for _, layer := range l {
		c.Layers = append(c.Layers, layer)
	}
}

// Compile the chain of layers into the model.
func (c *Chain) Compile(model Model) {
	for _, layer := range c.Layers {
		layer.Compile(model)
	}
}
