package env

import (
	"fmt"

	"github.com/pbarker/go-rl/pkg/v1/dense"
	sphere "github.com/pbarker/sphere/api/gen/go/v1alpha"
	"gorgonia.org/tensor"
)

// Normalizer will normalize the data coming from an environment.
type Normalizer interface {
	// Init initializes the normalizer.
	Init(env *Env) error

	// Norm normalizes the input data.
	Norm(input *tensor.Dense) *tensor.Dense
}

// MinMaxNormalizer is a min/max
type MinMaxNormalizer struct {
	min *tensor.Dense
	max *tensor.Dense
}

// NewMinMaxNormalizer returns a new min/max
func NewMinMaxNormalizer() *MinMaxNormalizer {
	return &MinMaxNormalizer{}
}

// Init the normalizer.
func (m *MinMaxNormalizer) Init(e *Env) (err error) {
	m.min, m.max, err = SpaceMinMax(e.GetObservationSpace())
	if err != nil {
		return
	}
	return
}

// Norm normalizes the input.
func (m *MinMaxNormalizer) Norm(input *tensor.Dense) *tensor.Dense {
	return dense.MinMaxNorm(input, m.min, m.max)
}

// EqWidthBinNormalizer is an EqWidthBinner applied using tensors.
type EqWidthBinNormalizer struct {
	intervals *tensor.Dense
	binner    *dense.EqWidthBinner
}

// NewEqWidthBinNormalizer is a new dense binner.
func NewEqWidthBinNormalizer(intervals *tensor.Dense) *EqWidthBinNormalizer {
	return &EqWidthBinNormalizer{intervals: intervals}
}

// Init the normalizer.
func (d *EqWidthBinNormalizer) Init(e *Env) error {
	min, max, err := SpaceMinMax(e.GetObservationSpace())
	if err != nil {
		return err
	}
	binner, err := dense.NewEqWidthBinner(d.intervals, min, max)
	if err != nil {
		return err
	}
	d.binner = binner
	return nil
}

// Norm normalizes the values placing them in their bins.
func (d *EqWidthBinNormalizer) Norm(input *tensor.Dense) (*tensor.Dense, error) {
	return d.binner.Bin(input)
}

// SpaceMinMax returns the min/max for a space as tensors.
func SpaceMinMax(space *sphere.Space) (min, max *tensor.Dense, err error) {
	switch s := space.GetInfo().(type) {
	case *sphere.Space_Box:
		shape := []int{}
		for _, i := range s.Box.GetShape() {
			shape = append(shape, int(i))
		}
		min = tensor.New(tensor.WithBacking(s.Box.GetLow()))
		max = tensor.New(tensor.WithBacking(s.Box.GetHigh()))
	case *sphere.Space_Discrete:
		min = tensor.New(tensor.WithBacking([]float32{0}))
		max = tensor.New(tensor.WithBacking([]float32{float32(s.Discrete.N)}))
	case *sphere.Space_MultiDiscrete:
		minB := []float32{}
		maxB := []float32{}
		for _, v := range s.MultiDiscrete.DiscreteSpaces {
			minB = append(minB, 0)
			maxB = append(maxB, float32(v))
		}
		min = tensor.New(tensor.WithBacking(minB))
		max = tensor.New(tensor.WithBacking(maxB))
	case *sphere.Space_StructSpace:
		err = fmt.Errorf("struct space not supported")
	default:
		err = fmt.Errorf("unknown action space type: %v", space)
	}
	return
}
