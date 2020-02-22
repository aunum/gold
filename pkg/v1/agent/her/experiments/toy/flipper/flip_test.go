package flipper

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"

	"github.com/pbarker/log"
)

func TestFlipper(t *testing.T) {
	for i := 0; i <= 2; i++ {
		log.Break()
		env := NewEnv(10)
		state, goal := env.Reset()
		log.Infov("state", state.Data())
		log.Infov("goal", goal.Data())

		for {
			action, isDone := pickRightAction(env)
			require.False(t, isDone)
			_, done, reward := env.Step(action)
			if done {
				require.Equal(t, 0, reward)
				break
			}
			require.Equal(t, -1, reward)
		}
		log.Infov("final state", env.state)
	}
	env := NewEnv(10)
	_, _ = env.Reset()
	var done bool
	for i := 0; i < env.MaxSteps(); i++ {
		_, done, _ = env.Step(0)
	}
	require.True(t, done)

	e2 := NewEnv(10, WithStatic())
	state1, goal1 := e2.Reset()
	s1 := state1.Clone().(*tensor.Dense)
	_, _, _ = e2.Step(0)
	state2, goal2 := e2.Reset()
	require.Equal(t, s1.Data(), state2.Data())
	require.Equal(t, goal1.Data(), goal2.Data())
}

func TestRandomBin(t *testing.T) {
	env := NewEnv(10)
	for i := 0; i < 100; i++ {
		log.Info(env.randomBinSlice())
	}
}

func pickRightAction(e *Env) (action int, done bool) {
	for i, v := range e.state {
		if e.goal[i] != v {
			return i, false
		}
	}
	return action, true
}
