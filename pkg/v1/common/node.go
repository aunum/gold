package common

import (
	g "gorgonia.org/gorgonia"
)

// Clip the node value.
func Clip(value *g.Node, min, max float64) (retVal *g.Node) {
	minNode := g.NewScalar(value.Graph(), g.Float64, g.WithValue(min), g.WithName("clip_min"))
	maxNode := g.NewScalar(value.Graph(), g.Float64, g.WithValue(max), g.WithName("clip_max"))

	minMask := g.Must(g.Lt(value, minNode, true))
	minVal := g.Must(g.HadamardProd(minNode, minMask))

	isMaskGt := g.Must(g.Gt(value, minNode, true))
	isMaskLt := g.Must(g.Lt(value, maxNode, true))
	isMask := g.Must(g.HadamardProd(isMaskGt, isMaskLt))
	isVal := g.Must(g.HadamardProd(value, isMask))

	maxMask := g.Must(g.Gt(value, maxNode, true))
	maxVal := g.Must(g.HadamardProd(maxNode, maxMask))

	retVal = g.Must(g.ReduceAdd(g.Nodes{minVal, isVal, maxVal}))
	return
}
