package track

import (
	"github.com/pbarker/logger"
	g "gorgonia.org/gorgonia"
)

// Value that has been tracked.
type Value interface {
	// Name of the value.
	Name() string

	// Scalar value.
	Scalar() float64

	// Print the value.
	Print()

	// Data converts the value to a historical value.
	Data(episode, timestep int) *HistoricalValue
}

// TrackedNodeValue is a tracked node value.
type TrackedNodeValue struct {
	// name of the tracked node value.
	name string

	// value of the tracked node value.
	value g.Value

	// index of the value.
	index int
}

// NewTrackedNodeValue returns a new tracked value.
func NewTrackedNodeValue(name string, index int) *TrackedNodeValue {
	var val g.Value
	return &TrackedNodeValue{
		name:  name,
		value: val,
		index: index,
	}
}

// Name of the value.
func (t *TrackedNodeValue) Name() string {
	return t.name
}

// Scalar value.
func (t *TrackedNodeValue) Scalar() float64 {
	data := t.value.Data()
	return toF64(data, t.index)
}

// Print the value.
func (t *TrackedNodeValue) Print() {
	logger.Infov(t.name, t.Scalar())
}

// Data converts the value to a historical value.
func (t *TrackedNodeValue) Data(episode, timestep int) *HistoricalValue {
	return &HistoricalValue{
		Name:     t.name,
		Value:    t.Scalar(),
		Timestep: timestep,
		Episode:  episode,
	}
}

// TrackedValue is a tracked value.
type TrackedValue struct {
	// name of the tracked value.
	name string

	// value of the tracked value.
	value interface{}

	// index of the scalar.
	index int
}

// NewTrackedValue returns a new tracked value.
func NewTrackedValue(name string, value interface{}, index int) *TrackedValue {
	return &TrackedValue{
		name:  name,
		value: value,
		index: index,
	}
}

// Name of the value.
func (t *TrackedValue) Name() string {
	return t.name
}

// Scalar value.
func (t *TrackedValue) Scalar() float64 {
	return toF64(t.value, t.index)
}

// Print the value.
func (t *TrackedValue) Print() {
	logger.Infov(t.name, t.Scalar())
}

// Data takes the current tracked value and returns a historical value.
func (t *TrackedValue) Data(episode, timestep int) *HistoricalValue {
	return &HistoricalValue{
		Name:     t.name,
		Value:    t.Scalar(),
		Timestep: timestep,
		Episode:  episode,
	}
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
