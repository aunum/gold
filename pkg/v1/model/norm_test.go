package model

import (
	"fmt"
	"testing"
)

func TestNorm(t *testing.T) {
	low := float32(-5.0)
	high := float32(5.0)
	x := float32(20.0)
	norm := MinMaxNorm(x, low, high)
	fmt.Println(norm)
}
