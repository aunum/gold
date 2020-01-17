package deepq

import (
	"testing"

	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestMemory(t *testing.T) {
	mem := NewMemory()
	s := tensor.New(tensor.WithShape(1, 4), tensor.WithBacking(tensor.Range(tensor.Float32, 0, 4)))
	ev := NewEvent(s, 0, &envv1.Outcome{Observation: s, Action: 1, Reward: 1.0, Done: false})

	for i := 0; i < 10; i++ {
		ev.Action = i
		mem.PushFront(ev)
	}

	sample, err := mem.Sample(5)
	require.NoError(t, err)

	require.Len(t, sample, 5)
}
