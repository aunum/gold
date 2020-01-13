package require

import (
	"github.com/pbarker/logger"
)

// Nil requires that the given value is nil or exists.
func Nil(v interface{}) {
	if v != nil {
		logger.Fatalf("%v must be nil", v)
	}
}

// NoError requires that the error is nil or exists.
func NoError(err error) {
	if err != nil {
		logger.Fatal(err)
	}
}
