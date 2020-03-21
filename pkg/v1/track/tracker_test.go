package track_test

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gorgonia.org/tensor"

	. "github.com/aunum/gold/pkg/v1/track"
	g "gorgonia.org/gorgonia"
)

func TestTracker(t *testing.T) {
	graph := g.NewGraph()
	tracker, err := NewTracker()
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
	fmt.Println("prod: ", prod)

	z := g.NewMatrix(graph, g.Float32, g.WithShape(prod.Shape()...), g.WithInit(g.RangedFrom(5)))

	sub := g.Must(g.Sub(z, prod))
	tracker.TrackValue("sub", sub)
	fmt.Println("sub: ", sub)

	vm := g.NewTapeMachine(graph)

	for ep := 0; ep < 100; ep++ {
		for ts := 0; ts < 10; ts++ {
			xB0 := tensor.New(tensor.WithShape(1, 5), tensor.WithBacking(tensor.Range(tensor.Float32, ep+ts, ep+ts+5)))
			g.Let(x, xB0)

			err := vm.RunAll()
			require.NoError(t, err)

			tracker.PrintAll()
			vm.Reset()

			err = tracker.LogStep(ep, ts)
			require.NoError(t, err)
		}
	}
	eph, err := tracker.GetHistory("prod")
	require.NoError(t, err)

	aggs := eph.Aggregate(Mean)
	fmt.Println("aggs: ", aggs)

	xys := aggs.GonumXYs()
	fmt.Println("xys: ", xys)

	plt, err := plot.New()
	require.NoError(t, err)
	line, err := plotter.NewLine(xys)
	require.NoError(t, err)
	plt.Add(line)
	fileName := "plot_test.png"
	err = plt.Save(3*vg.Inch, 4*vg.Inch, fileName)
	require.NoError(t, err)
}
