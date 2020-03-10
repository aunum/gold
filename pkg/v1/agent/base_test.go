package agent

import (
	"testing"
	"time"
)

func TestBase(t *testing.T) {
	base := NewBase("test")

	for ep := 0; ep < 100; ep++ {
		for ts := 0; ts < 10; ts++ {
			base.Tracker.TrackValue("test1", ep+ts)
			base.Tracker.TrackValue("test2", ep+ts+100)
			base.Tracker.LogStep(ep, ts)

		}
	}
	base.Serve()
	time.Sleep(60 * time.Second)
}
