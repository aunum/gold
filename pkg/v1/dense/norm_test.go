package dense

import (
	"fmt"
	"testing"

	"gorgonia.org/tensor"
)

func TestNorm(t *testing.T) {
	x := tensor.New(tensor.WithBacking([]float32{2.5, -3.0}))
	min := tensor.New(tensor.WithBacking([]float32{-5, -10}))
	max := tensor.New(tensor.WithBacking([]float32{5, 10}))
	norm := MinMaxNorm(x, min, max)
	fmt.Println("normalized: ", norm)
}
