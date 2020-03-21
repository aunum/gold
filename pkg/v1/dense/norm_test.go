package dense

import (
	"testing"

	"github.com/aunum/gold/pkg/v1/common/num"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestMinMax(t *testing.T) {
	xb := []float32{2.5, -3.0}
	x := tensor.New(tensor.WithBacking(xb))
	minB := []float32{-5, -10}
	min := tensor.New(tensor.WithBacking(minB))
	maxB := []float32{5, 10}
	max := tensor.New(tensor.WithBacking(maxB))

	norm, err := MinMaxNorm(x, min, max)
	require.NoError(t, err)

	fin := []float32{}
	for i, x := range xb {
		norm := num.MinMaxNorm(x, minB[i], maxB[i])
		fin = append(fin, norm)
	}
	require.Equal(t, fin, norm.Data().([]float32))
}

func TestMean(t *testing.T) {
	xb := []float32{2.5, -3.0, 5.0, 10.0}
	x := tensor.New(tensor.WithBacking(xb))
	m, err := Mean(x)
	require.NoError(t, err)
	require.Equal(t, m.GetF32(0), num.Mean(xb))
}

func TestStdDev(t *testing.T) {
	xb := []float32{2.5, -3.0, 5.0}
	x := tensor.New(tensor.WithBacking(xb))
	sigma, err := StdDev(x)
	require.NoError(t, err)
	require.Equal(t, sigma.GetF32(0), num.StdDev(xb))
}

func TestZNorm(t *testing.T) {
	xb := []float32{2.5, -3.0, 5.0}
	x := tensor.New(tensor.WithBacking(xb))

	norm, err := ZNorm(x)
	require.NoError(t, err)

	fin := []float32{}
	xMean := num.Mean(xb)
	xStdDev := num.StdDev(xb)
	for _, x := range xb {
		znorm := num.ZNorm(x, xMean, xStdDev)
		fin = append(fin, znorm)
	}
	require.Equal(t, fin, norm.Data().([]float32))
}
