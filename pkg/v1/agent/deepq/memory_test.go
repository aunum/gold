package deepq

import (
	"fmt"
	"testing"

	envv1 "github.com/aunum/gold/pkg/v1/env"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestMemory(t *testing.T) {
	mem := NewMemory()

	for i := 0; i < 10; i++ {
		s := tensor.New(tensor.WithShape(1, 4), tensor.WithBacking([]int{i, i + 1, i + 2, i + 3}))
		ev := NewEvent(s, 0, &envv1.Outcome{Observation: s, Action: i, Reward: 1.0, Done: false})
		mem.PushFront(ev)
	}

	sample, err := mem.Sample(5)
	require.NoError(t, err)
	for _, s := range sample {
		fmt.Printf("%#v\n", s.Outcome)
		fmt.Printf("%#v\n", s)
		fmt.Println("----")
	}
	require.Len(t, sample, 5)
}
