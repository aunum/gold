package model

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/guptarohit/asciigraph"
	"github.com/pbarker/logger"
	g "gorgonia.org/gorgonia"
)

// Tracker is a means of tracking values on a graph.
type Tracker struct {
	Values []*TrackedValue `json:"values"`

	graph    *g.ExprGraph
	encoder  *json.Encoder
	f        *os.File
	scanner  *bufio.Scanner
	filePath string
}

// TrackerOpt is a tracker option.
type TrackerOpt func(*Tracker)

// NewTracker returns a new tracker for a graph.
func NewTracker(graph *g.ExprGraph, opts ...TrackerOpt) (*Tracker, error) {
	f, err := ioutil.TempFile("", "stats.*.json")
	if err != nil {
		return nil, err
	}
	logger.Infof("tracking data in %s", f.Name())
	encoder := json.NewEncoder(f)
	scanner := bufio.NewScanner(f)
	t := &Tracker{
		Values:   []*TrackedValue{},
		graph:    graph,
		encoder:  encoder,
		f:        f,
		scanner:  scanner,
		filePath: f.Name(),
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

// TrackedValue is a tracked value.
type TrackedValue struct {
	Name  string
	Value g.Value
}

// Data takes the current tracked value and returns a historical value.
func (t *TrackedValue) Data() *HistoricalValue {
	return &HistoricalValue{
		Name:  t.Name,
		Value: t.Value.Data(),
	}
}

// Render the value.
func (t *TrackedValue) Render() {
	logger.Infov(t.Name, t.Value)
}

// History is the historical representation of a set of tracked values.
type History struct {
	// Values in the history.
	Values []*HistoricalValue `json:"values"`
}

// Get the value history with the given name.
func (h *History) Get(name string) *ValueHistory {
	vh := &ValueHistory{Name: name}
	history := []interface{}{}
	for _, value := range h.Values {
		if value.Name == name {
			history = append(history, value.Value)
		}
	}
	vh.History = history
	return vh
}

// HistoricalValue is a historical value.
type HistoricalValue struct {
	Name  string      `json:"name"`
	Value interface{} `json:"value"`
}

// Data yeilds the current tracked values into a historical structure.
func (t *Tracker) Data() *History {
	vals := []*HistoricalValue{}
	for _, v := range t.Values {
		vals = append(vals, v.Data())
	}
	return &History{vals}
}

// TrackValue tracks a nodes value.
func (t *Tracker) TrackValue(name string, node *g.Node) {
	tv := &TrackedValue{
		Name: name,
	}
	t.Values = append(t.Values, tv)
	g.Read(node, &tv.Value)
}

// Get a tracked value by name.
func (t *Tracker) Get(name string) (*TrackedValue, error) {
	for _, value := range t.Values {
		if value.Name == name {
			return value, nil
		}
	}
	return nil, fmt.Errorf("%q value does not exist", name)
}

// GetHistory gets all the history for a value.
func (t *Tracker) GetHistory(name string) (*ValueHistory, error) {
	values := []interface{}{}
	f, err := os.Open(t.filePath)
	if err != nil {
		return nil, err
	}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		b := scanner.Bytes()
		// fmt.Println("bytes :", b)

		var h History
		err := json.Unmarshal(b, &h)
		if err != nil {
			return nil, err
		}
		vh := h.Get(name)
		values = append(values, vh.History...)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return &ValueHistory{
		Name:    name,
		History: values,
	}, nil
}

// GetHistoryAll gets the history of all values.
func (t *Tracker) GetHistoryAll() ([]*ValueHistory, error) {
	all := []*ValueHistory{}
	for _, v := range t.Values {
		vh, err := t.GetHistory(v.Name)
		if err != nil {
			return nil, err
		}
		all = append(all, vh)
	}
	return all, nil
}

// Render a value.
func (t *Tracker) Render(name string) {
	v, err := t.Get(name)
	if err != nil {
		logger.Error(err)
	}
	v.Render()
}

// RenderAll values.
func (t *Tracker) RenderAll() {
	for _, value := range t.Values {
		value.Render()
	}
}

// Flush tracked values to store.
func (t *Tracker) Flush() error {
	return t.encoder.Encode(t.Data())
}

// ValueHistory is the history of a value.
type ValueHistory struct {
	Name    string
	History []interface{}
}

// Float32 converts the tracked values to a float32 slice.
func (v *ValueHistory) Float32() []float32 {
	ret := []float32{}
	for _, value := range v.History {
		ret = append(ret, value.(float32))
	}
	return ret
}

// Float64 converts the tracked values to a float64 slice.
func (v *ValueHistory) Float64() []float64 {
	ret := []float64{}
	for _, value := range v.History {
		ret = append(ret, value.(float64))
	}
	return ret
}

// Int converts the tracked values to a int slice.
func (v *ValueHistory) Int() []int {
	ret := []int{}
	for _, value := range v.History {
		ret = append(ret, value.(int))
	}
	return ret
}

// Int32 converts the tracked values to a int slice.
func (v *ValueHistory) Int32() []int32 {
	ret := []int32{}
	for _, value := range v.History {
		ret = append(ret, value.(int32))
	}
	return ret
}

// Int64 converts the tracked values to a int slice.
func (v *ValueHistory) Int64() []int64 {
	ret := []int64{}
	for _, value := range v.History {
		ret = append(ret, value.(int64))
	}
	return ret
}

// Render the tracked values.
func (v *ValueHistory) Render() {
	logger.Infov(v.Name, v.History)
}

// Chart the history.
func (v *ValueHistory) Chart() error {
	vals := v.toF64()
	if len(vals) == 0 {
		return fmt.Errorf("value history for %q is empty", v.Name)
	}
	graph := asciigraph.Plot(v.toF64(), asciigraph.Caption(v.Name))
	fmt.Println(graph)
	return nil
}

// toF64 will attempt to cast the underlying values to float64.
func (v *ValueHistory) toF64(index ...int) []float64 {
	ret := []float64{}
	if len(index) == 0 {
		index = []int{0}
	}
	for _, value := range v.History {
		switch val := value.(type) {
		case float64:
			ret = append(ret, val)
		case []float64:
			for _, i := range index {
				if len(val) < i+1 {
					logger.Fatal("index %v is not available in value %v", i, val)
				}
				ret = append(ret, val[i])
			}
		case float32:
			ret = append(ret, float64(val))
		case []float32:
			for _, i := range index {
				if len(val) < i+1 {
					logger.Fatal("index %v is not available in value %v", i, val)
				}
				ret = append(ret, float64(val[i]))
			}
		case int:
			ret = append(ret, float64(val))
		case []int:
			for _, i := range index {
				if len(val) < i+1 {
					logger.Fatal("index %v is not available in value %v", i, val)
				}
				ret = append(ret, float64(val[i]))
			}
		case int32:
			ret = append(ret, float64(val))
		case []int32:
			for _, i := range index {
				if len(val) < i+1 {
					logger.Fatal("index %v is not available in value %v", i, val)
				}
				ret = append(ret, float64(val[i]))
			}
		case int64:
			ret = append(ret, float64(val))
		case []int64:
			for _, i := range index {
				if len(val) < i+1 {
					logger.Fatal("index %v is not available in value %v", i, val)
				}
				ret = append(ret, float64(val[i]))
			}
		case []interface{}:
			vv := ValueHistory{Name: v.Name, History: val}
			ret = append(ret, vv.toF64()...)
		default:
			logger.Fatalf("unknown type %T %v could not cast to float64", val, val)
		}
	}
	return ret
}
