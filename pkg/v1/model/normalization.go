package model

// MinMaxNormalize is min-max normalization.
// https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
func MinMaxNormalize(x, min, max float32) float32 {
	return x - min/(max-min)
}

// MeanNormalize is mean normalization.
// https://en.wikipedia.org/wiki/Feature_scaling#Mean_normalization
func MeanNormalize(x, min, max, average float32) float32 {
	return x - average/max - min
}

// ZScoreNormalize uses z-score normalization.
// https://en.wikipedia.org/wiki/Feature_scaling#Standardization_(Z-score_Normalization)
func ZScoreNormalize() {

}

// UnitLengthNormalize implementes unit length normalization.
// https://en.wikipedia.org/wiki/Feature_scaling#Scaling_to_unit_length
func UnitLengthNormalize() {

}
