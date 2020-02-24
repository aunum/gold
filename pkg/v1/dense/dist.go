package dense

import (
	"math/rand"
	"reflect"
	"time"

	t "gorgonia.org/tensor"
)

// RandN generates a new dense tensor of the given shape with
// values populated from the standard normal distribution.
func RandN(dt t.Dtype, shape ...int) *t.Dense {
	size := t.Shape(shape).TotalSize()
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	switch dt.Kind() {
	case reflect.Int:
		backing := make([]int, size)
		for i := range backing {
			backing[i] = int(r.NormFloat64())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Int8:
		backing := make([]int8, size)
		for i := range backing {
			backing[i] = int8(r.NormFloat64())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Int16:
		backing := make([]int16, size)
		for i := range backing {
			backing[i] = int16(r.NormFloat64())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Int32:
		backing := make([]int32, size)
		for i := range backing {
			backing[i] = int32(r.NormFloat64())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Int64:
		backing := make([]int64, size)
		for i := range backing {
			backing[i] = int64(r.NormFloat64())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint:
		backing := make([]uint, size)
		for i := range backing {
			backing[i] = uint(r.Uint32())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint8:
		backing := make([]uint8, size)
		for i := range backing {
			backing[i] = uint8(r.Uint32())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint16:
		backing := make([]uint16, size)
		for i := range backing {
			backing[i] = uint16(r.Uint32())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint32:
		backing := make([]uint32, size)
		for i := range backing {
			backing[i] = uint32(r.Uint32())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Uint64:
		backing := make([]uint64, size)
		for i := range backing {
			backing[i] = uint64(r.Uint32())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Float32:
		backing := make([]float32, size)
		for i := range backing {
			backing[i] = float32(r.NormFloat64())
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Float64:
		backing := make([]float64, size)
		for i := range backing {
			backing[i] = rand.NormFloat64()
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Complex64:
		backing := make([]complex64, size)
		for i := range backing {
			backing[i] = complex(r.Float32(), float32(0))
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	case reflect.Complex128:
		backing := make([]complex128, size)
		for i := range backing {
			backing[i] = complex(r.Float64(), float64(0))
		}
		return t.New(t.WithShape(shape...), t.WithBacking(backing))
	default:
		panic("unkown type")
	}
}
