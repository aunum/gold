package common

// Mean of the values.
func Mean(vals []float64) float64 {
	l := float64(len(vals))
	var sum float64
	for _, val := range vals {
		sum += val
	}
	return sum / l
}
