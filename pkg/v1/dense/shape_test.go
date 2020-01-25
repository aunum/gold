package dense

import (
	"testing"

	"github.com/stretchr/testify/require"

	T "gorgonia.org/tensor"
)

func TestExpandDims(t *testing.T) {
	tens1 := T.New(T.WithShape(5), T.WithBacking(T.Range(T.Float32, 0, 5)))
	ExpandDims(tens1, 0)
	require.Equal(t, tens1.Shape(), T.Shape([]int{1, 5}))

	tens2 := T.New(T.WithShape(5), T.WithBacking(T.Range(T.Float32, 0, 5)))
	ExpandDims(tens2, 1)
	require.Equal(t, tens2.Shape(), T.Shape([]int{5, 1}))
}
