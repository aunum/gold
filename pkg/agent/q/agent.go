package q

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"time"

	"github.com/pbarker/go-rl/pkg/common"
	"github.com/pbarker/go-rl/pkg/space"
	"gorgonia.org/tensor"
)

// Agent that utilizes the Q-Learning algorithm.
type Agent struct {
	*Hyperparameters

	r           *rand.Rand
	actionSpace *space.Discrete
	table       Table
	minAlpha    float32
	minEpsilon  float32
}

// Hyperparameters for a Q-learning agent.
type Hyperparameters struct {
	// Epsilon is the rate at which the agent should explore vs exploit. The lower the value
	// the more exploitation.
	Epsilon float32

	// Gamma is the discount factor (0≤γ≤1). It determines how much importance we want to give to future
	// rewards. A high value for the discount factor (close to 1) captures the long-term effective award, whereas,
	// a discount factor of 0 makes our agent consider only immediate reward, hence making it greedy.
	Gamma float32

	// Alpha is the learning rate (0<α≤1). Just like in supervised learning settings, alpha is the extent
	// to which our Q-values are being updated in every iteration.
	Alpha float32

	// AdaDivisor is used in adaptive learning to tune the hyperparameters.
	AdaDivisor float32
}

// DefaultHyperparameters is the default agent configuration.
var DefaultHyperparameters = &Hyperparameters{
	Epsilon:    0.1,
	Gamma:      0.3,
	Alpha:      0.1,
	AdaDivisor: 5.0,
}

// NewAgent returns a new Q-learning agent. If table is nil, it will default to a basic in-memory table.
func NewAgent(h *Hyperparameters, actionSpaceSize int, table Table) *Agent {
	if table == nil || (reflect.ValueOf(table).Kind() == reflect.Ptr && reflect.ValueOf(table).IsNil()) {
		table = NewMemTable(actionSpaceSize)
	}
	s := rand.NewSource(time.Now().Unix())
	return &Agent{
		Hyperparameters: h,
		r:               rand.New(s),
		actionSpace:     space.NewDiscrete(actionSpaceSize),
		table:           table,
		minAlpha:        h.Alpha,
		minEpsilon:      h.Epsilon,
	}
}

// Adapt will adjust the hyperparameters based on th timestep.
func (a *Agent) Adapt(timestep int) {
	//max(self.min_alpha, min(1.0, 1.0 - math.log10((t + 1) / self.ada_divisor)))
	a.Epsilon = adapt(timestep, a.minEpsilon, a.AdaDivisor)
	fmt.Println("set epsilon to: ", a.Epsilon)
	a.Alpha = adapt(timestep, a.minAlpha, a.AdaDivisor)
	fmt.Println("set alpha to: ", a.Alpha)
}

func adapt(timestep int, min float32, ada float32) float32 {
	a := float32((timestep + 1)) / ada
	b := math.Log10(float64(a))
	c := 1.0 - b
	// fmt.Printf("a: %v b: %v c: %v\n", a, b, c)
	adapted := math.Min(1.0, c)
	max := math.Max(float64(min), adapted)
	// fmt.Printf("adapted: %v max: %v\n", adapted, max)
	return float32(max)
}

// Action returns the action that should be taken given the state hash.
func (a *Agent) Action(state *tensor.Dense) (int, error) {
	stateHash := HashState(state)
	var action int
	if common.RandFloat32(float32(0.0), float32(1.0)) < a.Epsilon {
		// explore
		action = a.actionSpace.Sample().(int)
	} else {
		// exploit
		var err error
		action, _, err = a.table.GetMax(stateHash)
		if err != nil {
			return 0, err
		}
	}
	return action, nil
}

// Learn using the Q-learning algorithm.
// Q(state,action)←(1−α)Q(state,action)+α(reward+γmaxaQ(next state,all actions))
func (a *Agent) Learn(action int, reward float32, state, nextState *tensor.Dense) error {
	stateHash := HashState(state)
	nextStateHash := HashState(nextState)
	oldVal, err := a.table.Get(stateHash, action)
	if err != nil {
		return err
	}
	_, nextMax, err := a.table.GetMax(nextStateHash)
	if err != nil {
		return err
	}

	// fmt.Printf("eq: oldVal %v + alpha %v * (reward %v + gamma %v * nextMax %v - oldVal %v)\n", oldVal, a.Alpha, reward, a.Gamma, nextMax, oldVal)
	// Q learning algorithm.
	newValue := (oldVal + a.Alpha) * (reward + a.Gamma*nextMax - oldVal)

	fmt.Printf("learning reward: %v on state: %v with new value: %v\n", reward, stateHash, newValue)
	err = a.table.Set(stateHash, action, newValue)
	if err != nil {
		return err
	}
	return nil
}

// Visualize the agents internal state.
func (a *Agent) Visualize() {
	a.table.Print()
}
