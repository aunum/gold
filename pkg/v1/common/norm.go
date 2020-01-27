package common

import (
	"github.com/pbarker/logger"
)

// MinMaxNorm is min-max normalization.
func MinMaxNorm(x, min, max float32) float32 {
	if x < min || x > max || min >= max {
		logger.Fatalf("parameters for min-max norm not valid")
	}
	return (x - min) / (max - min)
}

// MeanNorm is mean normalization.
func MeanNorm(x, min, max, average float32) float32 {
	if x < min || x > max || min >= max {
		logger.Fatalf("parameters for mean norm not valid")
	}
	return (x - average) / (max - min)
}

// ZNorm uses z-score normalization.
func ZNorm(x, mean, stdDev float32) float32 {
	return (x - mean) / stdDev
}
