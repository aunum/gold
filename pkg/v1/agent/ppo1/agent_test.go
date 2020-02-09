package ppo1

import (
	"fmt"
	"testing"
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
)

func TestDist(t *testing.T) {
	wieghts := []float64{0.01, 0.1, 0.89}

	for i := 0; i < 100; i++ {
		source := rand.NewSource(uint64(time.Now().UnixNano()))
		dist := distuv.NewCategorical(wieghts, source)
		r := dist.Rand()
		fmt.Println(r)
	}

}
