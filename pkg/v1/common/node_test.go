package common

import (
	"testing"

	"github.com/stretchr/testify/require"

	g "gorgonia.org/gorgonia"
)

func TestClip(t *testing.T) {
	for _, test := range []struct {
		val, min, max float64
		expected      float64
	}{
		{
			val:      3,
			min:      2,
			max:      4,
			expected: 3,
		},
		{
			val:      1,
			min:      2,
			max:      4,
			expected: 2,
		},
		{
			val:      5,
			min:      2,
			max:      4,
			expected: 4,
		},
		{
			val:      0,
			min:      -2,
			max:      2,
			expected: 0,
		},
		{
			val:      -1,
			min:      -2,
			max:      2,
			expected: -1,
		},
	} {
		graph := g.NewGraph()
		v := g.NewScalar(graph, g.Float64, g.WithValue(test.val), g.WithName("val"))
		ret := Clip(v, test.min, test.max)
		var retVal g.Value
		g.Read(ret, &retVal)

		vm := g.NewTapeMachine(graph)
		err := vm.RunAll()
		require.NoError(t, err)
		data := retVal.Data().(float64)
		require.Equal(t, test.expected, data)
	}
}
