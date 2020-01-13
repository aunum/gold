package common

import "math"

// Schedule is a means of transforming values based on timesteps.
type Schedule interface {
	// Value for the given step.
	Value(step int) float32
}

// ConstantSchedule just returns a constant value.
type ConstantSchedule struct {
	value float32
}

// NewConstantSchedule returns a new constant schedule.
func NewConstantSchedule(value float32) *ConstantSchedule {
	return &ConstantSchedule{
		value: value,
	}
}

// Value for the given step.
func (c *ConstantSchedule) Value(step int) float32 {
	return c.value
}

// LinearSchedule returns values on a linear means.
type LinearSchedule struct {
	// numTimesteps is the number of timesteps in the schedule.
	numTimesteps float64

	// initialValue for the schedule at the first timestep.
	initialValue float64

	// finalValue for the schedule at the last timestep.
	finalValue float64
}

// NewLinearSchedule returns a new LinearSchedule.
func NewLinearSchedule(numTimesteps int, initialValue, finalValue float32) *LinearSchedule {
	return &LinearSchedule{
		numTimesteps: float64(numTimesteps),
		initialValue: float64(initialValue),
		finalValue:   float64(finalValue),
	}
}

// Value for the given step.
func (l *LinearSchedule) Value(step int) float32 {
	fraction := math.Min(float64(step)/l.numTimesteps, 1.0)
	return float32(l.initialValue + fraction*(l.finalValue-l.initialValue))
}
