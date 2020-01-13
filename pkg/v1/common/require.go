package common

import (
	"github.com/pbarker/logger"
)

// RequireNil requires that the given value is nil.
func RequireNil(v interface{}) {
	if v != nil {
		logger.Fatalf("%v must be nil", v)
	}
}

// RequireNoError requires that the values doesn't have an error
func RequireNoError(err error) {
	if err != nil {
		logger.Fatal(err)
	}
}
