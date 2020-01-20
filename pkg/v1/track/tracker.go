package track

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/pbarker/logger"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	g "gorgonia.org/gorgonia"
)

// Tracker is a means of tracking values on a graph.
type Tracker struct {
	NodeValues []*TrackedNodeValue
	Values     []*TrackedValue

	Timestep int
	Episode  int

	encoder  *json.Encoder
	f        *os.File
	scanner  *bufio.Scanner
	filePath string
}

// TrackerOpt is a tracker option.
type TrackerOpt func(*Tracker)

// NewTracker returns a new tracker for a graph.
func NewTracker(opts ...TrackerOpt) (*Tracker, error) {
	f, err := ioutil.TempFile("", "stats.*.json")
	if err != nil {
		return nil, err
	}
	logger.Infof("tracking data in %s", f.Name())
	encoder := json.NewEncoder(f)
	scanner := bufio.NewScanner(f)
	t := &Tracker{
		NodeValues: []*TrackedNodeValue{},
		encoder:    encoder,
		f:          f,
		scanner:    scanner,
		filePath:   f.Name(),
	}
	return t, nil
}

// WithDir is a tracker option to set the directory in which logs are stored.
func WithDir(dir string) func(*Tracker) {
	return func(t *Tracker) {
		f, err := ioutil.TempFile(dir, "stats")
		if err != nil {
			logger.Fatal(err)
		}
		encoder := json.NewEncoder(f)
		t.encoder = encoder
		t.filePath = f.Name()
	}
}

// TrackedNodeValue is a tracked node value.
type TrackedNodeValue struct {
	// Name of the tracked node value.
	Name string

	// Value of the tracked node value.
	Value g.Value

	// Index of the value.
	Index int
}

// Data takes the current tracked value and returns a historical value.
func (t *TrackedNodeValue) Data(ts, episode int) *HistoricalValue {
	data := t.Value.Data()
	f := toF64(data, t.Index)
	return &HistoricalValue{
		Name:     t.Name,
		Value:    f,
		Timestep: ts,
		Episode:  episode,
	}
}

// Print the value.
func (t *TrackedNodeValue) Print() {
	logger.Infov(t.Name, t.Value)
}

// TrackedValue is a tracked value.
type TrackedValue struct {
	// Name of the tracked value.
	Name string

	// Value of the tracked value.
	Value float64
}

// Data takes the current tracked value and returns a historical value.
func (t *TrackedValue) Data(ts, episode int) *HistoricalValue {
	return &HistoricalValue{
		Name:     t.Name,
		Value:    t.Value,
		Timestep: ts,
		Episode:  episode,
	}
}

// Print the value.
func (t *TrackedValue) Print() {
	logger.Infov(t.Name, t.Value)
}

// History is the historical representation of a set of tracked values.
type History struct {
	// Values in the history.
	Values []*HistoricalValue `json:"values"`

	// Timestep of this history.
	Timestep int `json:"timestep"`

	// Episode of this history.
	Episode int `json:"episode"`
}

// Get the value history with the given name.
func (h *History) Get(name string) ValueHistory {
	history := ValueHistory{}
	for _, value := range h.Values {
		if value.Name == name {
			history = append(history, value)
		}
	}
	return history
}

// HistoricalValue is a historical value.
type HistoricalValue struct {
	// Name of the value.
	Name string `json:"name"`

	// Value of the value.
	Value float64 `json:"value"`

	// Timestep at which the value occured.
	Timestep int `json:"timestep"`

	// Episode at which the value occured.
	Episode int `json:"episode"`
}

// Data yeilds the current tracked values into a historical structure.
func (t *Tracker) Data() *History {
	vals := []*HistoricalValue{}
	for _, v := range t.Values {
		vals = append(vals, v.Data(t.Timestep, t.Episode))
	}
	for _, v := range t.NodeValues {
		vals = append(vals, v.Data(t.Timestep, t.Episode))
	}
	return &History{
		Values:   vals,
		Timestep: t.Timestep,
		Episode:  t.Episode,
	}
}

// IncTS increments the timestep of the tracker.
func (t *Tracker) IncTS() {
	t.Timestep++
}

// IncEpisode increments the epeisode of the tracker.
func (t *Tracker) IncEpisode() {
	t.Episode++
}

// TrackValue tracks a value.
func (t *Tracker) TrackValue(name string, value interface{}, index ...int) {
	i := 0
	if len(index) != 0 {
		i = index[0]
	}
	v := toF64(value, i)
	tv := &TrackedValue{
		Name:  name,
		Value: v,
	}
	t.Values = append(t.Values, tv)
}

// TrackNodeValue tracks a nodes value.
// An index can be provided if the value is non scalar.
// If multiple indexes are provided only the first one will be considered.
func (t *Tracker) TrackNodeValue(name string, node *g.Node, index ...int) {
	i := 0
	if len(index) != 0 {
		i = index[0]
	}
	tv := &TrackedNodeValue{
		Name:  name,
		Index: i,
	}
	t.NodeValues = append(t.NodeValues, tv)
	g.Read(node, &tv.Value)
}

// GetValue a tracked value by name.
func (t *Tracker) GetValue(name string) (*TrackedValue, error) {
	for _, value := range t.Values {
		if value.Name == name {
			return value, nil
		}
	}
	return nil, fmt.Errorf("%q value does not exist", name)
}

// GetNodeValue a tracked value by name.
func (t *Tracker) GetNodeValue(name string) (*TrackedNodeValue, error) {
	for _, value := range t.NodeValues {
		if value.Name == name {
			return value, nil
		}
	}
	return nil, fmt.Errorf("%q value does not exist", name)
}

// GetHistory gets all the history for a value.
func (t *Tracker) GetHistory(name string) (ValueHistory, error) {
	history := ValueHistory{}
	f, err := os.Open(t.filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		b := scanner.Bytes()

		var h History
		err := json.Unmarshal(b, &h)
		if err != nil {
			return nil, err
		}
		vh := h.Get(name)
		history = append(history, vh...)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return history, nil
}

// GetHistoryAll gets the history of all values.
func (t *Tracker) GetHistoryAll() ([]ValueHistory, error) {
	all := []ValueHistory{}
	for _, v := range t.Values {
		vh, err := t.GetHistory(v.Name)
		if err != nil {
			return nil, err
		}
		all = append(all, vh)
	}
	return all, nil
}

// PrintValue prints a value.
func (t *Tracker) PrintValue(name string) {
	v, err := t.GetValue(name)
	if err != nil {
		logger.Error(err)
	}
	v.Print()
}

// PrintNodeValue prints a value.
func (t *Tracker) PrintNodeValue(name string) {
	v, err := t.GetNodeValue(name)
	if err != nil {
		logger.Error(err)
	}
	v.Print()
}

// PrintAll values.
func (t *Tracker) PrintAll() {
	for _, value := range t.Values {
		value.Print()
	}
	for _, value := range t.NodeValues {
		value.Print()
	}
}

// Flush tracked values to store.
func (t *Tracker) Flush() error {
	return t.encoder.Encode(t.Data())
}

// Plot prints the charts for the tracker.
func (t *Tracker) Plot() error {
	vals, err := t.GetHistoryAll()
	if err != nil {
		return err
	}
	for _, val := range vals {
		err := val.Plot()
		if err != nil {
			return err
		}
	}
	return nil
}

// PrintHistoryAll prints the history of all values.
func (t *Tracker) PrintHistoryAll() error {
	vals, err := t.GetHistoryAll()
	if err != nil {
		return err
	}
	for _, val := range vals {
		val.Print()
	}
	return nil
}

// ValueHistory is the history of a value.
type ValueHistory []*HistoricalValue

// Print the tracked values.
func (v ValueHistory) Print() {
	logger.Infoy("history", v)
}

// Plot prints a chart of the history.
func (v ValueHistory) Plot() error {
	_, err := plot.New()
	if err != nil {
		return err
	}

	return nil
}

// Histories are a slice of history.
type Histories []*History

// EpisodeHistories is a history of episodes
type EpisodeHistories map[int]Histories

// GetEpisodeHistories returns the episode history.
func (t *Tracker) GetEpisodeHistories() (EpisodeHistories, error) {
	epHistories := EpisodeHistories{}
	f, err := os.Open(t.filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		b := scanner.Bytes()

		var h History
		err := json.Unmarshal(b, &h)
		if err != nil {
			return nil, err
		}

		epHistory, ok := epHistories[h.Episode]
		if !ok {
			epHistory = Histories{}
		}
		epHistory = append(epHistory, &h)
		epHistories[h.Episode] = epHistory
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return epHistories, nil
}

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

// HistoricalValues is a slice of historical value.
type HistoricalValues []*HistoricalValue

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
	return nil
}

func toF64(data interface{}, index int) float64 {
	var ret float64
	switch val := data.(type) {
	case float64:
		ret = val
	case []float64:
		ret = val[index]
	case float32:
		ret = float64(val)
	case []float32:
		ret = float64(val[index])
	case int:
		ret = float64(val)
	case []int:
		ret = float64(val[index])
	case int32:
		ret = float64(val)
	case []int32:
		ret = float64(val[index])
	case int64:
		ret = float64(val)
	case []int64:
		ret = float64(val[index])
	case []interface{}:
		ret = toF64(val[index], index)
	default:
		logger.Fatalf("unknown type %T %v could not cast to float64", val, val)
	}
	return ret
}
