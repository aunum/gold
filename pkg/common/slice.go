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

// RandFloat32 returns a random float in the given range.
func RandFloat32(min, max float32) float32 {
	return min + rand.Float32()*(max-min)
}

// MaxFloat32 returns the max index and max value for a float32 slice.
func MaxFloat32(vals []float32) (int, float32) {
	var max float32
	var maxI int
	for i, v := range vals {
		if i == 0 || v > max {
			max = v
			maxI = i
		}
	}
	return maxI, max
}
