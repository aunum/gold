package dense

import (
	"fmt"
	"reflect"

	t "gorgonia.org/tensor"
)

// Contains checks if a tensor contains a value.
func Contains(d *t.Dense, val interface{}) (contains bool, indicies []int) {
	iterator := d.Iterator()
	for i, err := iterator.Next(); err == nil; i, err = iterator.Next() {
		if val == d.Get(i) {
			contains = true
			indicies = append(indicies, i)
		}
	}
	return
}

// NormalizeZeros normalizes the zero values.
func NormalizeZeros(d *t.Dense) error {
	contains, indicies := Contains(d, ZeroValue(d.Dtype()))
	if !contains {
		return nil
	}
	for _, i := range indicies {
		d.Set(i, FauxZeroValue(d.Dtype()))
	}
	return nil
}

// ZeroValue for the given datatype.
func ZeroValue(dt t.Dtype) interface{} {
	switch dt.Kind() {
	case reflect.Int:
		return int(0)
	case reflect.Int8:
		return int8(0)
	case reflect.Int16:
		return int16(0)
	case reflect.Int32:
		return int32(0)
	case reflect.Int64:
		return int64(0)
	case reflect.Uint:
		return uint(0)
	case reflect.Uint8:
		return uint8(0)
	case reflect.Uint16:
		return uint16(0)
	case reflect.Uint32:
		return uint32(0)
	case reflect.Uint64:
		return uint64(0)
	case reflect.Float32:
		return float32(0)
	case reflect.Float64:
		return float64(0)
	case reflect.Complex64:
		return complex64(0)
	case reflect.Complex128:
		return complex128(0)
	default:
		panic(fmt.Sprintf("type not supported: %#v", dt))
	}
}

// FauxZero is the faux zero value used to prevent divde by zero errors.
const FauxZero = float64(1e-6)

// FauxZeroValue is a faux zero value for the given datatype.
func FauxZeroValue(dt t.Dtype) interface{} {
	switch dt.Kind() {
	case reflect.Float32:
		return float32(FauxZero)
	case reflect.Float64:
		return float64(FauxZero)
	case reflect.Complex64:
		return complex64(FauxZero)
	case reflect.Complex128:
		return complex128(FauxZero)
	default:
		panic(fmt.Sprintf("type not supported: %#v", dt))
	}
}
