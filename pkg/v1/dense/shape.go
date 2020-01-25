package dense

import (
	"gorgonia.org/tensor"
)

// ExpandDims expands the dimensions of a tensor along the given axis.
func ExpandDims(t *tensor.Dense, axis int) error {
	dims := []int{}
	if axis == 0 {
		dims = append(dims, 1)
		dims = append(dims, t.Shape()...)

	} else {
		dims = append(dims, t.Shape()...)
		dims = append(dims, 1)
	}
	err := t.Reshape(dims...)
	return err
}
