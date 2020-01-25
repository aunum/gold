package common

import (
	"fmt"
	"testing"
)

func TestSchedule(t *testing.T) {
	s := DefaultDecaySchedule()
	for i := 0; i <= 100; i++ {
		fmt.Println(s.Value())
	}
}
