package q

import (
	"fmt"
	"github.com/pbarker/go-rl/pkg/space"
	"github.com/schwarmco/go-cartesian-product"
	"gorgonia.org/tensor"
	"hash/fnv"
)

// Table is the qualtiy table which stores the quality of an action by state.
type Table interface {
}

// MemTable is an in memory Table with a row for every state, and a column for every action. State is
// held as a hash of observations.
type MemTable struct {
	table map[uint32]*tensor.Dense
}

// NewMemTable returns a new MemTable with the dimensions defined by the observation and
// action space sizes.
func NewMemTable(actionSpace space.Space, stateTable StateTable) Table {
	return &MemTable{
		table:      tensor.New(tensor.WithShape(), tensor.Of(tensor.Int)),
		stateTable: stateTable,
	}
}

// StateTable holds the Cartesian Product of all possible states.
type StateTable interface {
	// N returns the total number of possible states.
	N() int
}

// MemStateTable is an in memory implementation of a StateTable. It is a map of hash
// to observation values.
type MemStateTable struct {
}

// NewMemStateTable creates a new StateTable. Each array given assumes the possible values
// for that observation are discrete. Returns a Cartesian Product of all possible states.
func NewMemStateTable(observations ...[]interface{}) StateTable {
	ch := cartesian.Iter(observations...)
	stateTable := map[uint32][]interface{}{}
	for observations := range ch {
		h := Hash(observations)
		stateTable[h] = observations
	}
	return &MemStateTable{table: stateTable}
}

// N returns the total number of possible states.
func (m *MemStateTable) N() int {
	return len(m.table)
}

// Hash observations into an integer value. Note: this requires observations to always
// occur in a specific order.
func Hash(observations interface{}) uint32 {
	h := fnv.New32a()
	s := fmt.Sprintf("%v", observations)
	h.Write([]byte(s))
	return h.Sum32()
}
