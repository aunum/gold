package dense

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
	T "gorgonia.org/tensor"
)

func TestMax(t *testing.T) {
	t0 := tensor.New(tensor.WithBacking([]float32{5, 8, 3, 2}))
	max, err := AMax(t0, 0)
	require.NoError(t, err)
	m := max.(float32)
	require.Equal(t, m, float32(8))
}

func TestConcat(t *testing.T) {
	t0 := T.New(T.WithShape(1, 5), T.WithBacking(T.Range(T.Float32, 0, 5)))
	t1 := T.New(T.WithShape(1, 5), T.WithBacking(T.Range(T.Float32, 5, 10)))
	t2 := T.New(T.WithShape(1, 5), T.WithBacking(T.Range(T.Float32, 10, 15)))

	c, err := Concat(0, t0, t1, t2)
	require.NoError(t, err)
	fmt.Println("final\n", c)

	tf := T.New(T.WithShape(3, 5), T.WithBacking(T.Range(T.Float32, 0, 15)))
	require.True(t, c.Eq(tf))
}
