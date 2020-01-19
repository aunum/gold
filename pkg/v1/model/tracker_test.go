package model_test

import (
	"fmt"
	"testing"

	"gorgonia.org/tensor"

	"github.com/stretchr/testify/require"

	. "github.com/pbarker/go-rl/pkg/v1/model"
	g "gorgonia.org/gorgonia"
)

func TestTracker(t *testing.T) {
	graph := g.NewGraph()
	tracker, err := NewTracker(graph)
	require.NoError(t, err)

	xB := tensor.New(tensor.WithShape(1, 5), tensor.WithBacking(tensor.Random(tensor.Float32, 5)))
	fmt.Println(xB)
	x := g.NewMatrix(graph, g.Float32, g.WithValue(xB))
	fmt.Println("x: ", x.Value())

	yB := tensor.New(tensor.WithShape(5, 1), tensor.WithBacking(tensor.Random(tensor.Float32, 5)))
	fmt.Println(yB)
	y := g.NewMatrix(graph, g.Float32, g.WithValue(yB))
	fmt.Println("y: ", y.Value())

	prod := g.Must(g.Mul(x, y))
	tracker.TrackValue("prod", prod)

	z := g.NewMatrix(graph, g.Float32, g.WithShape(prod.Shape()...), g.WithInit(g.RangedFrom(5)))

	sub := g.Must(g.Sub(z, prod))
	tracker.TrackValue("sub", sub)

	vm := g.NewTapeMachine(graph)

	for i := 0; i < 100; i++ {
		xB0 := tensor.New(tensor.WithShape(1, 5), tensor.WithBacking(tensor.Range(tensor.Float32, i, i+5)))
		g.Let(x, xB0)

		err := vm.RunAll()
		require.NoError(t, err)

		tracker.RenderAll()
		vm.Reset()

		err = tracker.Flush()
		require.NoError(t, err)
	}
	tvs, err := tracker.GetHistoryAll()
	require.NoError(t, err)
	for _, tv := range tvs {
		err = tv.Chart()
		require.NoError(t, err)
	}
}
