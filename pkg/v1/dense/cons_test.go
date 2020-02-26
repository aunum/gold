package dense

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFill(t *testing.T) {
	v := float32(5)
	ret := Fill(v, 3, 4)
	iterator := ret.Iterator()
	for i, err := iterator.Next(); err == nil; i, err = iterator.Next() {
		r := ret.GetF32(i)
		require.Equal(t, v, r)
	}
}
