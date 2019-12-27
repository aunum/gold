package common

import "math/rand"

// MakeIntRange creates an int slice.
func MakeIntRange(min, max int) []int {
	a := make([]int, max-min+1)
	for i := range a {
		a[i] = min + i
	}
	return a
}

// RandFloat64 returns a random float in the given range.
func RandFloat64(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

// MaxFloat64 returns the max index and max value for a float32 slice.
func MaxFloat64(vals []float64) (int, float64) {
	var max float64
	var maxI int
	for i, v := range vals {
		if i == 0 || v > max {
			max = v
			maxI = i
		}
	}
	return maxI, max
}
