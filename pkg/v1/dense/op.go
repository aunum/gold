package dense

import (
	"fmt"
	"reflect"

	"gorgonia.org/tensor"
	t "gorgonia.org/tensor"
)

// Div safely divides 'a' by 'b' by slightly augmenting any zero values in 'b'.
//
// TODO: check efficiency, may be cheaper to just always add a small value.
func Div(a, b *t.Dense) (*t.Dense, error) {
	err := NormalizeZeros(b)
	if err != nil {
		return nil, err
	}
	return a.Div(b)
}

// Neg negates the tensor elementwize.
func Neg(v *tensor.Dense) (*t.Dense, error) {
	ret, err := BroadcastMul(v, tensor.New(tensor.FromScalar(NegValue(v.Dtype()))))
	if err != nil {
		return nil, err
	}
	return ret, nil
}

// NegValue for the given datatype.
func NegValue(dt t.Dtype) interface{} {
	switch dt.Kind() {
	case reflect.Int:
		return int(-1)
	case reflect.Int8:
		return int8(-1)
	case reflect.Int16:
		return int16(-1)
	case reflect.Int32:
		return int32(-1)
	case reflect.Int64:
		return int64(-1)
	case reflect.Float32:
		return float32(-1)
	case reflect.Float64:
		return float64(-1)
	case reflect.Complex64:
		return complex64(-1)
	case reflect.Complex128:
		return complex128(-1)
	default:
		panic(fmt.Sprintf("type not supported: %#v", dt))
	}
}
