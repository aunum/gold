package q

import (
	"math/rand"
	"time"

	tensor "gorgonia.org/tensor"
)

// Agent in environment.
type Agent struct {
	// the learning rate is how much you accept the new value over the old value.
	alpha float32

	// the discount factor, this balances the immediate and future rewards.
	gamma float32

	

	// QualityTable is a reference for our agent to select the best action based on the Q value.
	QualityTable *tensor.Dense
}

// NewAgent returns a new agent.
func NewAgent() *Agent {
	t := tensor.New(tensor.WithShape(2, 2))
	return &Agent{
		QualityTable: t,
	}
}

// Action performs an action.
func (a *Agent) Action() {
	// percent you want to explore
	epsilon := 0.2

	s1 := rand.NewSource(time.Now().UnixNano())
	r1 := rand.New(s1)

	min := 0.0
	max := 1.0
	r := min + r1.Float64()*(max-min)

	if r < epsilon {
		// explore using a random action.

	} else {
		// exploit selecting the action with the max value.

	}
}
