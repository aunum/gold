package model

import (
	"fmt"

	"github.com/pbarker/logger"
	g "gorgonia.org/gorgonia"
)

// Tracker is a means of tracking values on a graph.
type Tracker struct {
	graph  *g.ExprGraph
	values []*TrackedValue
}

// NewTracker returns a new tracker for a graph.
func NewTracker(graph *g.ExprGraph) *Tracker {
	return &Tracker{
		graph:  graph,
		values: []*TrackedValue{},
	}
}

// TrackedValue is a tracked value.
type TrackedValue struct {
	name  string
	value g.Value
	node  *g.Node
}

// Render the value.
func (t *TrackedValue) Render() {
	logger.Infoy(t.name, t.value)
}

// TrackValue tracks a nodes value.
func (t *Tracker) TrackValue(name string, node *g.Node) {
	var v g.Value
	g.Read(node, &v)
	tv := &TrackedValue{
		name:  name,
		value: v,
		node:  node,
	}
	t.values = append(t.values, tv)
}

// Get a tracked value by name.
func (t *Tracker) Get(name string) (*TrackedValue, error) {
	for _, value := range t.values {
		if value.name == name {
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
	for _, value := range t.values {
		value.Render()
	}
}
