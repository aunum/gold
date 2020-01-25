package dense

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestMax(t *testing.T) {
	t0 := tensor.New(tensor.WithBacking([]float32{5, 8, 3, 2}))
	max, err := AMax(t0, 0)
	require.NoError(t, err)
	m := max.(float32)
	require.Equal(t, m, float32(8))
}
