package dense

import (
	"fmt"

	t "gorgonia.org/tensor"
)

// MinMaxNorm normalizes the input x using pointwise min-max normalization along the axis.
// This is a pointwise operation and requires the shape of the min/max tensors be equal to x.
func MinMaxNorm(x, min, max *t.Dense) (*t.Dense, error) {
	if !min.Shape().Eq(x.Shape()) {
		return nil, fmt.Errorf("min shape %v must match x shape %v", min.Shape(), x.Shape())
	}
	if !max.Shape().Eq(x.Shape()) {
		return nil, fmt.Errorf("max shape %v must match x shape %v", max.Shape(), x.Shape())
	}

	// y=(x-min)/(max-min)
	ret, err := x.Sub(min)
	if err != nil {
		return nil, err
	}
	space, err := max.Sub(min)
	if err != nil {
		return nil, err
	}
	ret, err = ret.Div(space)
	if err != nil {
		return nil, err
	}
	return ret, nil
}

// ZNorm normalizes x using z-score normalization along the axis.
// y=x-mean/stddev
func ZNorm(x *t.Dense, along ...int) (*t.Dense, error) {
	if len(along) == 0 {
		along = []int{0}
	}
	axis := along[0]
	mu, err := Mean(x, axis)
	if err != nil {
		return nil, err
	}
	mus, err := mu.Repeat(0, x.Shape()[axis])
	if err != nil {
		return nil, err
	}
	ret, err := x.Sub(mus.(*t.Dense))
	if err != nil {
		return nil, err
	}
	sigma, err := StdDev(x, axis)
	if err != nil {
		return nil, err
	}
	sigmas, err := sigma.Repeat(0, x.Shape()[axis])
	if err != nil {
		return nil, err
	}
	ret, err = ret.Div(sigmas.(*t.Dense))
	if err != nil {
		return nil, err
	}
	return ret, nil
}

// Mean of the tensor along the axis.
func Mean(x *t.Dense, along ...int) (*t.Dense, error) {
	if len(along) == 0 {
		along = []int{0}
	}
	axis := along[0]
	sum, err := x.Sum(axis)
	if err != nil {
		return nil, err
	}
	if len(x.Shape()) < axis {
		return nil, fmt.Errorf("tensor shape %v does not contain the axis %v", x.Shape(), along)
	}

	// TODO: how to not cast this?
	size := t.New(t.WithBacking([]float32{float32(x.Shape()[axis])}))
	mean, err := sum.Div(size)
	if err != nil {
		return nil, err
	}
	return mean, nil
}

// StdDev is the standard deviation of the tensor along the axis.
// y=sqrt(sum((x-mu)^2)/n)
func StdDev(x *t.Dense, along ...int) (*t.Dense, error) {
	if len(along) == 0 {
		along = []int{0}
	}
	axis := along[0]
	mu, err := Mean(x, axis)
	if err != nil {
		return nil, err
	}
	mus, err := mu.Repeat(0, x.Shape()[axis])
	if err != nil {
		return nil, err
	}
	distance, err := x.Sub(mus.(*t.Dense))
	if err != nil {
		return nil, err
	}

	abs, err := t.Square(distance)
	if err != nil {
		return nil, err
	}
	sum, err := abs.(*t.Dense).Sum(0)
	if err != nil {
		return nil, err
	}

	// TODO: how to not cast this? or just standardize on f32?
	size := t.New(t.WithBacking([]float32{float32(x.Shape()[axis])}))
	inner, err := sum.Div(size)
	if err != nil {
		return nil, err
	}
	ret, err := t.Sqrt(inner)
	if err != nil {
		return nil, err
	}

	return ret.(*t.Dense), nil
}
