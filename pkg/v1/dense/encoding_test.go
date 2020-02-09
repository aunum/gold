package dense

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestOneHotVector(t *testing.T) {
	v, err := OneHotVector(2, 6, tensor.Float32)
	require.NoError(t, err)

	require.Equal(t, []float32{0, 0, 1, 0, 0, 0}, v.Data().([]float32))
}
