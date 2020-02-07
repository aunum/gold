package dense

import (
	"fmt"

	t "gorgonia.org/tensor"
)

// Concat a list of tensors along a given axis.
func Concat(axis int, tensors ...*t.Dense) (retVal *t.Dense, err error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("no tensors provided to concat")
	}
	retVal = tensors[0]
	retVal, err = retVal.Concat(axis, tensors[1:]...)
	return
}
