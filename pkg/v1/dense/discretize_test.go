package dense_test

import (
	"fmt"
	"testing"

	"github.com/aunum/gold/pkg/v1/dense"
	"github.com/stretchr/testify/require"
	. "gorgonia.org/tensor"
)

func TestDenseEqWidthBinner(t *testing.T) {
	low := New(WithShape(1, 4), WithBacking([]float32{-12.0, -0.5, -5.0, 0}))
	high := New(WithShape(1, 4), WithBacking([]float32{12, 0.5, 5.0, 10.0}))
	intervals := New(WithShape(1, 4), WithBacking(Range(Int, 4, 8)))
	binner, err := dense.NewEqWidthBinner(intervals, low, high)
	require.Nil(t, err)

	observation := New(WithShape(1, 4), WithBacking([]float32{-8.0, -0.2, 2.0, 5.0}))
	binned, err := binner.Bin(observation)
	require.Nil(t, err)

	fmt.Printf("binned: %v \n", binned)

	outHighObservation := New(WithShape(1, 4), WithBacking([]float32{20.0, -0.2, 2.0, 5.0}))
	_, err = binner.Bin(outHighObservation)
	require.NotNil(t, err, "should have errored on out of high bounds")

	outLowObservation := New(WithShape(1, 4), WithBacking([]float32{-20.0, -0.2, 2.0, 5.0}))
	_, err = binner.Bin(outLowObservation)
	require.NotNil(t, err, "should have errored on out of low bounds")
}
