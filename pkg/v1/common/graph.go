package common

import (
	"fmt"
	"io/ioutil"
	"os/exec"

	"github.com/pbarker/go-rl/pkg/v1/common/require"
	"github.com/pbarker/log"

	g "gorgonia.org/gorgonia"
)

// Visualize the graph.
func Visualize(graph *g.ExprGraph) {
	f, err := ioutil.TempFile("", "graph.*.dot")
	require.NoError(err)
	_, err = f.Write([]byte(graph.ToDot()))
	require.NoError(err)
	tempPath := f.Name()
	svgPath := fmt.Sprintf("%s.svg", f.Name())
	log.Info("saved file: ", tempPath)
	cmd := exec.Command("dot", "-Tsvg", tempPath, "-O")
	err = cmd.Run()
	require.NoError(err)
	cmd = exec.Command("open", svgPath)
	err = cmd.Run()
	require.NoError(err)
}
