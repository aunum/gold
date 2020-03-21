// Package common provides common methods and tooling for the agents.
package common

import (
	"fmt"
	"time"

	"github.com/aunum/log"
)

// Retry callback function for the number of attempts sleeping for the duration in-between.
func Retry(attempts int, sleep time.Duration, callback func() error) (err error) {
	for i := 1; ; i++ {
		err = callback()
		if err == nil {
			return
		}
		if i >= attempts {
			break
		}
		time.Sleep(sleep)
		log.Errorf("retrying after error: %v", err)
	}
	return fmt.Errorf("after %d attempts, last error: %s", attempts, err)
}
