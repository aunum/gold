package dense_test

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"

	. "github.com/pbarker/go-rl/pkg/v1/dense"
	T "gorgonia.org/tensor"
)

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
