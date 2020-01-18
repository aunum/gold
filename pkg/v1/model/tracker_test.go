package model_test

import (
	"testing"

	. "github.com/pbarker/go-rl/pkg/v1/model"
	g "gorgonia.org/gorgonia"
)

func TestTracker(t *testing.T) {
	graph := g.NewGraph()
	NewTracker(graph)
}
