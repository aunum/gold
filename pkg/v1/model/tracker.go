package model

import (
	"encoding/json"
	"fmt"
	"io/ioutil"

	"github.com/pbarker/logger"
	g "gorgonia.org/gorgonia"
)

// Tracker is a means of tracking values on a graph.
type Tracker struct {
	Values []*TrackedValue `json:"values"`

	graph   *g.ExprGraph
	encoder *json.Encoder
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
	t := &Tracker{
		Values:  []*TrackedValue{},
		graph:   graph,
		encoder: encoder,
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
