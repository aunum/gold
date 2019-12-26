package q_test

import (
	"testing"

	. "github.com/pbarker/go-rl/pkg/algo/q"
	"github.com/pbarker/go-rl/pkg/common"
)

// func TestQ(t *testing.T) {
// 	// s, err := sphere.NewLocalServer(sphere.GymServerConfig)
// 	// require.Nil(t, err)

// 	// env, err := s.Make("CartPole-v0")
// 	// require.Nil(t, err)
// }

func TestAgent(t *testing.T) {
	// Test Cartpole-v0

	// Create observations.
	theta := common.MakeIRange(0, 6)
	thetaDot := common.MakeIRange(0, 12)

	// Create State Table
	stateTable := NewMemStateTable(theta, thetaDot)

	// Create actions.
	actions := common.MakeIRange(0, 3)

	// Create a new QTable
	NewMemTable := NewMemTable(actions, stateTable)
}
