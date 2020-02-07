package op

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
		ret, err := Clip(v, test.min, test.max)
		require.NoError(t, err)
		var retVal g.Value
		g.Read(ret, &retVal)

		vm := g.NewTapeMachine(graph)
		err = vm.RunAll()
		require.NoError(t, err)
		data := retVal.Data().(float64)
		require.Equal(t, test.expected, data)
	}
}

func TestMin(t *testing.T) {
	for _, test := range []struct {
		a, b     float64
		expected float64
	}{
		{
			a:        1,
			b:        2,
			expected: 1,
		},
		{
			a:        -5,
			b:        -3,
			expected: -5,
		},
		{
			a:        -1,
			b:        0,
			expected: -1,
		},
		{
			a:        5,
			b:        5,
			expected: 5,
		},
	} {
		graph := g.NewGraph()
		a := g.NewScalar(graph, g.Float64, g.WithValue(test.a), g.WithName("a"))
		b := g.NewScalar(graph, g.Float64, g.WithValue(test.b), g.WithName("b"))
		ret, err := Min(a, b)
		require.NoError(t, err)
		var retVal g.Value
		g.Read(ret, &retVal)

		vm := g.NewTapeMachine(graph)
		err = vm.RunAll()
		require.NoError(t, err)
		data := retVal.Data().(float64)
		require.Equal(t, test.expected, data)
	}
}

func TestMax(t *testing.T) {
	for _, test := range []struct {
		a, b     float64
		expected float64
	}{
		{
			a:        1,
			b:        2,
			expected: 2,
		},
		{
			a:        -5,
			b:        -3,
			expected: -3,
		},
		{
			a:        -1,
			b:        0,
			expected: 0,
		},
		{
			a:        5,
			b:        5,
			expected: 5,
		},
	} {
		graph := g.NewGraph()
		a := g.NewScalar(graph, g.Float64, g.WithValue(test.a), g.WithName("a"))
		b := g.NewScalar(graph, g.Float64, g.WithValue(test.b), g.WithName("b"))
		ret, err := Max(a, b)
		require.NoError(t, err)
		var retVal g.Value
		g.Read(ret, &retVal)

		vm := g.NewTapeMachine(graph)
		err = vm.RunAll()
		require.NoError(t, err)
		data := retVal.Data().(float64)
		require.Equal(t, test.expected, data)
	}
}
