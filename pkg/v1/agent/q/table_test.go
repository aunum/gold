package q_test

import (
	"testing"

	. "github.com/aunum/gold/pkg/v1/agent/q"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestMemTable(t *testing.T) {
	actionSpaceSize := 6

	table := NewMemTable(actionSpaceSize)

	observation1 := tensor.New(tensor.WithShape(2, 4), tensor.WithBacking(tensor.Range(tensor.Float32, 0, 8)))
	qVal1 := float32(0.5)
	action1 := 0
	state1 := HashState(observation1)
	err := table.Set(state1, action1, qVal1)
	require.Nil(t, err)

	qRes1, err := table.Get(state1, action1)
	require.Equal(t, qVal1, qRes1)
	require.NoError(t, err)

	observation2 := tensor.New(tensor.WithShape(2, 4), tensor.WithBacking(tensor.Range(tensor.Float32, 8, 16)))
	qVal2 := float32(0.2)
	action2 := 1
	state2 := HashState(observation2)
	err = table.Set(state2, action2, qVal2)
	require.Nil(t, err)

	qRes2, err := table.Get(state2, action2)
	require.Equal(t, qVal2, qRes2)
	require.NoError(t, err)

	err = table.Set(state1, action2, qVal2)
	require.NoError(t, err)

	action, qval, err := table.GetMax(state1)
	require.Nil(t, err)
	require.Equal(t, action, 0)
	require.Equal(t, qval, float32(0.5))
}
