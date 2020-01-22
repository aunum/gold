package track

import (
	"fmt"

	"gonum.org/v1/plot/plotter"
)

// AggregatorName is the name of an aggregator.
type AggregatorName string

const (
	// MeanAggregatorName is the name of the mean aggregator.
	MeanAggregatorName AggregatorName = "mean"

	// ModeAggregatorName is the name of the mode aggregator.
	ModeAggregatorName AggregatorName = "mode"

	// MaxAggregatorName is the name of the max aggregator.
	MaxAggregatorName AggregatorName = "max"
)

// AggregatorNames holds all of the current aggregator names.
var AggregatorNames = []AggregatorName{MeanAggregatorName, ModeAggregatorName, MaxAggregatorName}

// AggregatorFromName returns an aggregator from its name.
func AggregatorFromName(name string) (Aggregator, error) {
	switch AggregatorName(name) {
	case MeanAggregatorName:
		return MeanAggregator, nil
	case ModeAggregatorName:
		return ModeAggregator, nil
	case MaxAggregatorName:
		return MaxAggregator, nil
	default:
		return nil, fmt.Errorf("aggregator %q unknown", name)
	}
}

// EpisodeHistories is a history of episodes
type EpisodeHistories map[int]Histories

// Aggregator aggregates historical values into a single value.
type Aggregator func(HistoricalValues) float64

// MeanAggregator returns the mean of the historical values.
func MeanAggregator(vals HistoricalValues) float64 {
	l := float64(len(vals))
	var sum float64
	for _, val := range vals {
		sum += val.TrackedValue
	}
	return sum / l
}

// MaxAggregator returns the max of the historical values.
func MaxAggregator(vals HistoricalValues) float64 {
	var max float64
	for _, val := range vals {
		if val.TrackedValue > max {
			max = val.TrackedValue
		}
	}
	return max
}

// ModeAggregator returns the max of the historical values.
func ModeAggregator(vals HistoricalValues) float64 {
	modes := map[float64]int{}
	for _, val := range vals {
		v, ok := modes[val.TrackedValue]
		if !ok {
			v = 0
		}
		modes[val.TrackedValue] = v + 1
	}
	var maxI int
	var maxV float64
	for val, i := range modes {
		if i > maxI {
			maxV = val
		}
	}
	return maxV
}

// EpisodeAggregates is a map of episode to aggregated values.
type EpisodeAggregates map[int]float64

// Aggregate histories returning a map of episode to aggregated values.
func (e EpisodeHistories) Aggregate(name string, aggregator Aggregator) EpisodeAggregates {
	epAggs := EpisodeAggregates{}
	for episode, histories := range e {
		epVals := HistoricalValues{}
		for _, history := range histories {
			for _, value := range history.Values {
				if value.Name == name {
					epVals = append(epVals, value)
				}
			}
		}
		aggregated := aggregator(epVals)
		epAggs[episode] = aggregated
	}
	return epAggs
}

// GonumXYs returns the episode aggregates as gonum xy pairs.
func (e EpisodeAggregates) GonumXYs() plotter.XYs {
	xys := plotter.XYs{}
	for episode, value := range e {
		xy := plotter.XY{
			X: float64(episode),
			Y: value,
		}
		xys = append(xys, xy)
	}
	return xys
}

// ChartjsXY conforms to the expected point data structure for chartjs charts.
type ChartjsXY struct {
	X float64 `json:"x"`
	Y float64 `json:"y"`
}

// ChartjsXYs conforms to the expected set of point data structure for chartjs charts.
type ChartjsXYs []ChartjsXY

// ChartjsXYs returns the episode aggregates as gonum xy pairs.
func (e EpisodeAggregates) ChartjsXYs() ChartjsXYs {
	xys := ChartjsXYs{}
	for episode, value := range e {
		xy := ChartjsXY{
			X: float64(episode),
			Y: value,
		}
		xys = append(xys, xy)
	}
	return xys.Order()
}

// Order the xys by x.
func (c ChartjsXYs) Order() ChartjsXYs {
	xys := make([]ChartjsXY, len(c))
	for _, xy := range c {
		xys[int(xy.X)] = xy
	}
	return xys
}
