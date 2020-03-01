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
			a:         tensor.New(tensor.FromScalar(float32(2))),
			b:         tensor.New(tensor.FromScalar(float32(1))),
			expected:  tensor.New(tensor.FromScalar(float32(3))),
			shouldErr: false,
		},
		{
			name:      "scalar vector",
			a:         tensor.New(tensor.FromScalar(float32(1))),
			b:         Fill(float32(3), 5),
			expected:  Fill(float32(9), 5),
			shouldErr: false,
		},
		{
			name:      "scalar matrix one",
			a:         tensor.New(tensor.FromScalar(float32(1))),
			b:         Fill(float32(3), 1, 5),
			expected:  Fill(float32(9), 1, 5),
			shouldErr: false,
		},
		{
			name:      "scalar matrix",
			a:         tensor.New(tensor.FromScalar(float32(1))),
			b:         Fill(float32(3), 2, 5),
			expected:  Fill(float32(9), 2, 5),
			shouldErr: false,
		},
		{
			name:      "scalar tensor",
			a:         tensor.New(tensor.FromScalar(float32(1))),
			b:         Fill(float32(3), 3, 2, 5),
			expected:  Fill(float32(9), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "vector vector",
			a:         Fill(float32(3), 5),
			b:         Fill(float32(3), 5),
			expected:  Fill(float32(9), 5),
			shouldErr: false,
		},
		{
			name:      "matrix vector one",
			a:         Fill(float32(3), 1, 5),
			b:         Fill(float32(3), 5),
			expected:  Fill(float32(9), 1, 5),
			shouldErr: false,
		},
		{
			name:      "matrix vector",
			a:         Fill(float32(3), 2, 5),
			b:         Fill(float32(3), 5),
			expected:  Fill(float32(9), 2, 5),
			shouldErr: false,
		},
		{
			name:      "matrix vector fail",
			a:         Fill(float32(3), 4),
			b:         Fill(float32(3), 2, 5),
			expected:  nil,
			shouldErr: true,
		},
		{
			name:      "matrix matrix",
			a:         Fill(float32(3), 2, 5),
			b:         Fill(float32(3), 2, 5),
			expected:  Fill(float32(9), 2, 5),
			shouldErr: false,
		},
		{
			name:      "matrix matrix one",
			a:         Fill(float32(3), 1, 5),
			b:         Fill(float32(3), 2, 5),
			expected:  Fill(float32(9), 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor vector",
			a:         Fill(float32(3), 3, 2, 5),
			b:         Fill(float32(3), 5),
			expected:  Fill(float32(9), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor vector fail",
			a:         Fill(float32(3), 3, 2, 4),
			b:         Fill(float32(3), 5),
			expected:  nil,
			shouldErr: true,
		},
		{
			name:      "tensor matrix",
			a:         Fill(float32(3), 3, 2, 5),
			b:         Fill(float32(3), 2, 5),
			expected:  Fill(float32(9), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor matrix fail",
			a:         Fill(float32(3), 3, 3, 5),
			b:         Fill(float32(3), 2, 5),
			expected:  nil,
			shouldErr: true,
		},
		{
			name:      "tensor tensor",
			a:         Fill(float32(3), 3, 2, 5),
			b:         Fill(float32(3), 3, 2, 5),
			expected:  Fill(float32(9), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor tensor one",
			a:         Fill(float32(3), 3, 1, 5),
			b:         Fill(float32(3), 3, 2, 5),
			expected:  Fill(float32(9), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor tensor fail",
			a:         Fill(float32(3), 3, 4, 5),
			b:         Fill(float32(3), 3, 2, 5),
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

func TestBroadcastMul(t *testing.T) {
	for _, test := range []struct {
		name      string
		a         *tensor.Dense
		b         *tensor.Dense
		expected  *tensor.Dense
		shouldErr bool
	}{
		{
			name:      "scalar scalar",
			a:         tensor.New(tensor.FromScalar(float32(2))),
			b:         tensor.New(tensor.FromScalar(float32(3))),
			expected:  tensor.New(tensor.FromScalar(float32(6))),
			shouldErr: false,
		},
		{
			name:      "scalar vector",
			a:         tensor.New(tensor.FromScalar(float32(2))),
			b:         Fill(float32(3), 5),
			expected:  Fill(float32(6), 5),
			shouldErr: false,
		},
		{
			name:      "scalar matrix one",
			a:         tensor.New(tensor.FromScalar(float32(2))),
			b:         Fill(float32(3), 1, 5),
			expected:  Fill(float32(6), 1, 5),
			shouldErr: false,
		},
		{
			name:      "scalar matrix",
			a:         tensor.New(tensor.FromScalar(float32(2))),
			b:         Fill(float32(3), 2, 5),
			expected:  Fill(float32(6), 2, 5),
			shouldErr: false,
		},
		{
			name:      "scalar tensor",
			a:         tensor.New(tensor.FromScalar(float32(2))),
			b:         Fill(float32(3), 3, 2, 5),
			expected:  Fill(float32(6), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "vector vector",
			a:         Fill(float32(3), 5),
			b:         Fill(float32(3), 5),
			expected:  Fill(float32(9), 5),
			shouldErr: false,
		},
		{
			name:      "matrix vector one",
			a:         Fill(float32(3), 1, 5),
			b:         Fill(float32(3), 5),
			expected:  Fill(float32(9), 1, 5),
			shouldErr: false,
		},
		{
			name:      "matrix vector",
			a:         Fill(float32(3), 2, 5),
			b:         Fill(float32(3), 5),
			expected:  Fill(float32(9), 2, 5),
			shouldErr: false,
		},
		{
			name:      "matrix vector fail",
			a:         Fill(float32(3), 4),
			b:         Fill(float32(3), 2, 5),
			expected:  nil,
			shouldErr: true,
		},
		{
			name:      "matrix matrix",
			a:         Fill(float32(3), 2, 5),
			b:         Fill(float32(3), 2, 5),
			expected:  Fill(float32(9), 2, 5),
			shouldErr: false,
		},
		{
			name:      "matrix matrix one",
			a:         Fill(float32(3), 1, 5),
			b:         Fill(float32(3), 2, 5),
			expected:  Fill(float32(9), 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor vector",
			a:         Fill(float32(3), 3, 2, 5),
			b:         Fill(float32(3), 5),
			expected:  Fill(float32(9), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor vector fail",
			a:         Fill(float32(3), 3, 2, 4),
			b:         Fill(float32(3), 5),
			expected:  nil,
			shouldErr: true,
		},
		{
			name:      "tensor matrix",
			a:         Fill(float32(3), 3, 2, 5),
			b:         Fill(float32(3), 2, 5),
			expected:  Fill(float32(9), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor matrix fail",
			a:         Fill(float32(3), 3, 3, 5),
			b:         Fill(float32(3), 2, 5),
			expected:  nil,
			shouldErr: true,
		},
		{
			name:      "tensor tensor",
			a:         Fill(float32(3), 3, 2, 5),
			b:         Fill(float32(3), 3, 2, 5),
			expected:  Fill(float32(9), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor tensor one",
			a:         Fill(float32(3), 3, 1, 5),
			b:         Fill(float32(3), 3, 2, 5),
			expected:  Fill(float32(9), 3, 2, 5),
			shouldErr: false,
		},
		{
			name:      "tensor tensor fail",
			a:         Fill(float32(3), 3, 4, 5),
			b:         Fill(float32(3), 3, 2, 5),
			expected:  nil,
			shouldErr: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			ret, err := BroadcastMul(test.a, test.b)
			if test.shouldErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.True(t, ret.Eq(test.expected))

			ret, err = BroadcastMul(test.b, test.a)
			if test.shouldErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.True(t, ret.Eq(test.expected))
		})
	}
}
