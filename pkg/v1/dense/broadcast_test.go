package dense

import (
	"testing"

	"github.com/stretchr/testify/require"

	"gorgonia.org/tensor"
)

func TestBroadcastAdd(t *testing.T) {
	for _, test := range []struct {
		name      string
		a         *tensor.Dense
		b         *tensor.Dense
		expected  *tensor.Dense
		shouldErr bool
	}{
		{
			name:      "scalar scalar",
			a:         tensor.Ones(tensor.Float32, 5),
			b:         tensor.Ones(tensor.Float32, 5),
			expected:  Fill(float32(2), 5),
			shouldErr: false,
		},
		{
			name:      "matrix scalar one",
			a:         tensor.Ones(tensor.Float32, 1, 5),
			b:         tensor.Ones(tensor.Float32, 5),
			expected:  Fill(float32(2), 1, 5),
			shouldErr: false,
		},
		{
			name:      "matrix scalar",
			a:         tensor.Ones(tensor.Float32, 2, 5),
			b:         tensor.Ones(tensor.Float32, 5),
			expected:  Fill(float32(2), 2, 5),
			shouldErr: false,
		},
		{
			name:      "matrix scalar fail",
			a:         tensor.Ones(tensor.Float32, 4),
			b:         tensor.Ones(tensor.Float32, 2, 5),
			expected:  nil,
			shouldErr: true,
		},
		{
			name:      "matrix matrix",
			a:         tensor.Ones(tensor.Float32, 2, 5),
			b:         tensor.Ones(tensor.Float32, 2, 5),
			expected:  Fill(float32(2), 2, 5),
			shouldErr: false,
		},
		{
			name:      "matrix matrix one",
			a:         tensor.Ones(tensor.Float32, 1, 5),
			b:         tensor.Ones(tensor.Float32, 2, 5),
			expected:  Fill(float32(2), 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor scalar",
			a:         tensor.Ones(tensor.Float32, 3, 2, 5),
			b:         tensor.Ones(tensor.Float32, 5),
			expected:  Fill(float32(2), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor scalar fail",
			a:         tensor.Ones(tensor.Float32, 3, 2, 4),
			b:         tensor.Ones(tensor.Float32, 5),
			expected:  nil,
			shouldErr: true,
		},
		{
			name:      "tensor matrix",
			a:         tensor.Ones(tensor.Float32, 3, 2, 5),
			b:         tensor.Ones(tensor.Float32, 2, 5),
			expected:  Fill(float32(2), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor matrix fail",
			a:         tensor.Ones(tensor.Float32, 3, 3, 5),
			b:         tensor.Ones(tensor.Float32, 2, 5),
			expected:  nil,
			shouldErr: true,
		},
		{
			name:      "tensor tensor",
			a:         tensor.Ones(tensor.Float32, 3, 2, 5),
			b:         tensor.Ones(tensor.Float32, 3, 2, 5),
			expected:  Fill(float32(2), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor tensor one",
			a:         tensor.Ones(tensor.Float32, 3, 1, 5),
			b:         tensor.Ones(tensor.Float32, 3, 2, 5),
			expected:  Fill(float32(2), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor tensor fail",
			a:         tensor.Ones(tensor.Float32, 3, 4, 5),
			b:         tensor.Ones(tensor.Float32, 3, 2, 5),
			expected:  nil,
			shouldErr: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			ret, err := BroadcastAdd(test.a, test.b)
			if test.shouldErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.True(t, ret.Eq(test.expected))

			ret, err = BroadcastAdd(test.b, test.a)
			if test.shouldErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.True(t, ret.Eq(test.expected))
		})
	}
}
