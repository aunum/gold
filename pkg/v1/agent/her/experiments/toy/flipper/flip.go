package flipper

import (
	"math/rand"

	"github.com/pbarker/log"
	"gorgonia.org/tensor"
)

// Env is the env.
type Env struct {
	state       []float32
	goal        []float32
	steps       int
	N           int
	static      bool
	staticState []float32
	staticGoal  []float32
}

// EnvOpt is an env option.
type EnvOpt func(*Env)

// WithStatic sets the observation and goal to static values.
func WithStatic() func(*Env) {
	return func(e *Env) {
		e.static = true
	}
}

// NewEnv returns a new environment.
func NewEnv(n int, opts ...EnvOpt) *Env {
	e := &Env{
		N: n,
	}
	for _, opt := range opts {
		opt(e)
	}
	e.state = e.randomBinSlice()
	e.goal = e.randomBinSlice()
	if e.static {
		e.staticState = copyFloat(e.state)
		e.staticGoal = copyFloat(e.goal)
	}
	return e
}

// Step the env.
func (e *Env) Step(action int) (observation *tensor.Dense, done bool, reward int) {
	e.steps++
	if e.state[action] == float32(0.0) {
		e.state[action] = float32(1.0)
	} else {
		e.state[action] = float32(0.0)
	}
	equal := sliceEqual(e.state, e.goal)
	if equal {
		reward = 0
	} else {
		reward = -1
	}
	done = equal
	if e.steps >= e.N {
		done = true
	}
	return tensor.New(tensor.WithShape(1, 10), tensor.WithBacking(e.state)), done, reward
}

// Reset the env.
func (e *Env) Reset() (state, goal *tensor.Dense) {
	log.Info("resetting env")
	e.state = e.randomBinSlice()
	e.goal = e.randomBinSlice()
	if e.static {
		e.state = copyFloat(e.staticState)
		e.goal = e.staticGoal
	}
	if sliceEqual(e.state, e.goal) {
		log.Warning("recusively calling reset due to equal slices")
		e.Reset()
	}
	e.steps = 0
	return tensor.New(tensor.WithShape(1, e.N), tensor.WithBacking(e.state)), tensor.New(tensor.WithShape(1, e.N), tensor.WithBacking(e.goal))
}

// MaxSteps are the max steps.
func (e *Env) MaxSteps() int {
	return e.N
}

func (e *Env) randomBinSlice() (ret []float32) {
	for i := 0; i < e.N; i++ {
		ret = append(ret, float32(rand.Intn(2)))
	}
	return
}

func sliceEqual(a, b []float32) bool {
	for i, v := range a {
		if b[i] != v {
			return false
		}
	}
	return true
}

func copyFloat(v []float32) []float32 {
	ret := []float32{}
	for _, val := range v {
		ret = append(ret, val)
	}
	return ret
}
