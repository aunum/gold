package track

import (
	"encoding/json"
	"net/http"

	"gonum.org/v1/plot/plotter"
)

// EpisodeHistories is a history of episodes
type EpisodeHistories map[int]Histories

// Aggregator aggregates historical values into a single value.
type Aggregator func(HistoricalValues) float64

// MeanAggregator returns the mean of the historical values by their potential indicies.
func MeanAggregator(vals HistoricalValues) float64 {
	l := float64(len(vals))
	var sum float64
	for _, val := range vals {
		sum += val.Value
	}
	return sum / l
}

// AggregatedValues is a map of name of historical values to their aggregations.
type AggregatedValues map[string]float64

// EpisodeAggregates is a map of episode to aggregated values.
type EpisodeAggregates map[int]AggregatedValues

// Aggregate histories returning a map of episode to aggregated values.
func (e EpisodeHistories) Aggregate(aggregator Aggregator) EpisodeAggregates {
	epAggs := EpisodeAggregates{}
	for episode, histories := range e {
		allVals := map[string]HistoricalValues{}
		for _, history := range histories {
			for _, value := range history.Values {
				hv, ok := allVals[value.Name]
				if !ok {
					hv = HistoricalValues{}
				}
				allVals[value.Name] = append(hv, value)
			}
		}
		agVals := AggregatedValues{}
		for name, values := range allVals {
			aggregated := aggregator(values)
			agVals[name] = aggregated
		}
		epAggs[episode] = agVals
	}
	return epAggs
}

// XYs runs over every episode and aggregates the timestep values using the aggregator and returns
// a map of value name to aggregated episodic values.
func (e EpisodeAggregates) XYs() map[string]plotter.XYs {
	xys := map[string]plotter.XYs{}
	for episode, values := range e {
		for name, value := range values {
			vals, ok := xys[name]
			if !ok {
				vals = plotter.XYs{}
			}
			v := plotter.XY{
				X: float64(episode),
				Y: value,
			}
			vals = append(vals, v)
			xys[name] = vals
		}
	}
	return xys
}

// AggregateHandler is an HTTP handler for the tracker serving aggregates.
func (t *Tracker) AggregateHandler(w http.ResponseWriter, req *http.Request) {
	h, err := t.GetEpisodeHistories()
	if err != nil {
		w.Write([]byte(err.Error()))
		w.WriteHeader(500)
		return
	}
	aggs := h.Aggregate(MeanAggregator)
	xys := aggs.XYs()
	b, err := json.Marshal(xys)
	if err != nil {
		w.Write([]byte(err.Error()))
		w.WriteHeader(500)
		return
	}
	w.Write(b)
	w.WriteHeader(200)
}
