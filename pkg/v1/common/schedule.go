package common

import "math"

// Schedule is a means of transforming values based on timesteps.
type Schedule interface {
	// Value for the given step.
	Value() float32

	// Initial value
	Initial() float32
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
func (c *ConstantSchedule) Value() float32 {
	return c.value
}

// Initial value.
func (c *ConstantSchedule) Initial() float32 {
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

	currentStep int
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
func (l *LinearSchedule) Value() float32 {
	fraction := math.Min(float64(l.currentStep)/l.numTimesteps, 1.0)
	l.currentStep++
	return float32(l.initialValue + fraction*(l.finalValue-l.initialValue))
}

// Initial value for the schedule.
func (l *LinearSchedule) Initial() float32 {
	return float32(l.initialValue)
}

// DefaultLinearSchedule returns a linear schedule with some sensible defaults.
func DefaultLinearSchedule(numTimesteps int) *LinearSchedule {
	return NewLinearSchedule(numTimesteps, 1.0, 0.1)
}

// DecaySchedule returns values on an exponential decay means.
type DecaySchedule struct {
	// decay is the amount the value should decay at each step.
	decayRate float64

	// initialValue for the schedule at the first timestep.
	initialValue float64

	// minValue for the schedule at the last timestep.
	minValue float64

	currentValue float64
}

// NewDecaySchedule returns a new DecaySchedule.
func NewDecaySchedule(decayRate, initialValue, minValue float32) *DecaySchedule {
	return &DecaySchedule{
		decayRate:    float64(decayRate),
		initialValue: float64(initialValue),
		minValue:     float64(minValue),
		currentValue: float64(initialValue),
	}
}

// Value for the given step. Will decay with each call.
func (d *DecaySchedule) Value() float32 {
	d.currentValue *= d.decayRate
	d.currentValue = math.Max(d.minValue, d.currentValue)
	return float32(d.currentValue)
}

// Initial value for the schedule.
func (d *DecaySchedule) Initial() float32 {
	return float32(d.initialValue)
}

// DecayScheduleOpt is an option for a decay schedule.
type DecayScheduleOpt func(*DecaySchedule)

// WithDecayRate adds a decay rate to a default decay schedule.
func WithDecayRate(rate float32) func(*DecaySchedule) {
	return func(d *DecaySchedule) {
		d.decayRate = float64(rate)
	}
}

// WithMinValue adds a minimum value rate to a default decay schedule.
func WithMinValue(rate float32) func(*DecaySchedule) {
	return func(d *DecaySchedule) {
		d.decayRate = float64(rate)
	}
}

// DefaultDecaySchedule is the default decay schedule.
func DefaultDecaySchedule(opts ...DecayScheduleOpt) *DecaySchedule {
	s := &DecaySchedule{
		decayRate:    0.995,
		initialValue: 1.0,
		minValue:     0.01,
		currentValue: 1.0,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}
