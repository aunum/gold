package dense

import "gorgonia.org/tensor"

// MinMaxNorm normalizes the input x using min-max normalization.
func MinMaxNorm(x, min, max *tensor.Dense) *tensor.Dense {
	iterator := x.Iterator()
	backing := []float32{}
	for i, err := iterator.Next(); err == nil; i, err = iterator.Next() {
		x := x.GetF32(i)
		min := min.GetF32(i)
		max := max.GetF32(i)
		xPrime := (x - min) / (max - min)
		backing = append(backing, xPrime)
	}
	out := tensor.New(tensor.WithBacking(backing))
	return out
}
