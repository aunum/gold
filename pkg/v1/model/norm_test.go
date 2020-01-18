package model

import (
	"fmt"
	"math"
	"testing"
)

func TestNorm(t *testing.T) {
	low := float32(math.Inf(-1))
	high := float32(math.Inf(1))
	x := float32(20.0)
	norm := MinMaxNorm(x, low, high)
	fmt.Println(norm)
}
