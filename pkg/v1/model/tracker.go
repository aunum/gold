package model

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/pbarker/logger"
	g "gorgonia.org/gorgonia"
)

// Tracker is a means of tracking values on a graph.
type Tracker struct {
	Values TrackedValues `json:"values"`

	graph   *g.ExprGraph
	encoder *json.Encoder
	f       *os.File
	scanner *bufio.Scanner
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
		Values:  []*TrackedValue{},
		graph:   graph,
		encoder: encoder,
		f:       f,
		scanner: scanner,
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
	}
}

// TrackedValue is a tracked value.
type TrackedValue struct {
	Name  string  `json:"name"`
	Value g.Value `json:"value"`
}

// Render the value.
func (t *TrackedValue) Render() {
	logger.Infov(t.Name, t.Value)
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
func (t *Tracker) GetHistory(name string) ([]*TrackedValue, error) {
	values := []*TrackedValue{}
	for t.scanner.Scan() {
		var t *Tracker
		err := json.Unmarshal(t.scanner.Bytes(), t)
		if err != nil {
			return nil, err
		}
		tv, err := t.Get(name)
		if err != nil {
			return nil, err
		}
		values = append(values, tv)
	}
	if err := t.scanner.Err(); err != nil {
		return nil, err
	}
	return values, nil
}

// RenderValue renders a value.
func (t *Tracker) RenderValue(name string) {
	v, err := t.Get(name)
	if err != nil {
		logger.Error(err)
	}
	v.Render()
}

// Render all values.
func (t *Tracker) Render() {
	for _, value := range t.Values {
		value.Render()
	}
}

// Flush tracked values to store.
func (t *Tracker) Flush() error {
	return t.encoder.Encode(t)
}

// TrackedValues is a slice of tracked value.
type TrackedValues []*TrackedValue

// Float32 converts the tracked values to a float32 slice.
func (t TrackedValues) Float32() []float32 {
	ret := []float32{}
	for _, value := range t {
		ret = append(ret, value.Value.Data().(float32))
	}
	return ret
}

// Float64 converts the tracked values to a float64 slice.
func (t TrackedValues) Float64() []float64 {
	ret := []float64{}
	for _, value := range t {
		ret = append(ret, value.Value.Data().(float64))
	}
	return ret
}

// Int converts the tracked values to a int slice.
func (t TrackedValues) Int() []int {
	ret := []int{}
	for _, value := range t {
		ret = append(ret, value.Value.Data().(int))
	}
	return ret
}

// Int32 converts the tracked values to a int slice.
func (t TrackedValues) Int32() []int32 {
	ret := []int32{}
	for _, value := range t {
		ret = append(ret, value.Value.Data().(int32))
	}
	return ret
}

// Int64 converts the tracked values to a int slice.
func (t TrackedValues) Int64() []int64 {
	ret := []int64{}
	for _, value := range t {
		ret = append(ret, value.Value.Data().(int64))
	}
	return ret
}

// Data converts returns an interface of the values.
func (t TrackedValues) Data() []interface{} {
	ret := []interface{}{}
	for _, value := range t {
		ret = append(ret, value.Value.Data())
	}
	return ret
}

// Name of the tracked value.
func (t TrackedValues) Name() string {
	if len(t) == 0 {
		logger.Fatal("tracked values is empty")
	}
	return t[0].Name
}

// Render the tracked values.
func (t TrackedValues) Render() {
	logger.Infov(t.Name(), t.Data())
}
