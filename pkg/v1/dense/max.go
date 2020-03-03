package dense

import (
	"fmt"

	"gorgonia.org/tensor"
)

// AMax returns the maximum value in a tensor along an axis.
// TODO: support returning slice.
func AMax(d *tensor.Dense, axis int) (interface{}, error) {
	maxIndex, err := d.Argmax(axis)
	if err != nil {
		return nil, err
	}
	max := d.Get(maxIndex.GetI(0))
	return max, nil
}

// AMaxF32 returns the maximum value in a tensor along an axis as a float32.
func AMaxF32(d *tensor.Dense, axis int) (float32, error) {
	max, err := AMax(d, axis)
	if err != nil {
		return 0, err
	}
	f, ok := max.(float32)
	if !ok {
		return f, fmt.Errorf("could not cast %v to float32", max)
	}
	return f, nil
}
