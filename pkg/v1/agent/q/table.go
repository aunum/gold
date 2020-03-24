package q

import (
	"fmt"
	"hash/fnv"

	"github.com/aunum/gold/pkg/v1/common/num"
	"github.com/aunum/log"
	"github.com/k0kubun/pp"
	"gorgonia.org/tensor"
)

// Table is the qualtiy table which stores the quality of an action by state.
type Table interface {
	// GetMax returns the action with the max Q value for a given state hash.
	GetMax(state uint32) (action int, qValue float32, err error)

	// Get the Q value for the given state and action.
	Get(state uint32, action int) (float32, error)

	// Set the q value of the action taken for a given state.
	Set(state uint32, action int, value float32) error

	// Clear the table.
	Clear() error

	// Pretty print the table.
	Print()
}

// MemTable is an in memory Table with a row for every state, and a column for every action. State is
// held as a hash of observations.
type MemTable struct {
	actionSpaceSize int
	table           map[uint32][]float32
}

// NewMemTable returns a new MemTable with the dimensions defined by the observation and
// action space sizes.
func NewMemTable(actionSpaceSize int) Table {
	return &MemTable{
		actionSpaceSize: actionSpaceSize,
		table:           map[uint32][]float32{},
	}
}

// GetMax returns the action with the max Q value for a given state hash.
func (m *MemTable) GetMax(state uint32) (action int, qValue float32, err error) {
	qv, ok := m.table[state]
	if !ok {
		log.Debug("state does not exist yet: ", state)
		return 0, 0.0, nil
	}
	// fmt.Println("state exists! ", state)
	action, qValue = num.MaxF32(qv)
	return
}

// Get the Q value for the given state and action.
func (m *MemTable) Get(state uint32, action int) (float32, error) {
	qv, ok := m.table[state]
	if !ok {
		return 0.0, nil
	}
	if len(qv) < action+1 {
		return 0.0, fmt.Errorf("action %d outside of action space size %d", action, m.actionSpaceSize)
	}
	return qv[action], nil
}

// Set the quality of the action taken for a given state.
func (m *MemTable) Set(state uint32, action int, qValue float32) error {
	qv, ok := m.table[state]
	if !ok {
		qv = make([]float32, m.actionSpaceSize)
	}
	qv[action] = qValue
	m.table[state] = qv
	return nil
}

// Clear the table.
func (m *MemTable) Clear() error {
	m.table = map[uint32][]float32{}
	return nil
}

// Print the table with a pretty printer.
func (m *MemTable) Print() {
	for state, values := range m.table {
		fmt.Println("-----")
		fmt.Printf("---\nstate: %d\nqvalues: %s\n", state, pp.Sprint(values))
		fmt.Println("-----")
	}
}

// HashState observations into an integer value. Note: this requires observations to always
// occur in the same order.
func HashState(observations *tensor.Dense) uint32 {
	h := fnv.New32a()
	s := fmt.Sprintf("%v", observations)
	h.Write([]byte(s))
	return h.Sum32()
}
