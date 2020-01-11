package deepq

import (
	"github.com/gammazero/deque"
)

// Memory for the dqn agent.
type Memory struct {
	deque.Deque
}

// NewMemory returns a new Memory store.
func NewMemory() *Memory {
	return &Memory{
		Deque: deque.Deque{},
	}
}

// Sample from the memory with the given batch size.
func (m *Memory) Sample(batchsize int) {

}
