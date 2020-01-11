package errors

import (
	"github.com/pbarker/sphere/pkg/common/logger"
)

// Require checks if the error is nil, if not it logs it an exits with code 1.
func Require(err error) {
	if err != nil {
		logger.Fatal(err)
	}
}
