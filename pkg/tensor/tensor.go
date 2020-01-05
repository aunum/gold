package tensor

import (
	"fmt"

	"gorgonia.org/tensor"
)

// ToFloat32 will attempt to cast the given tensor int values to float32 vals.
func ToFloat32(t *tensor.Dense) (*tensor.Dense, error) {
	new := tensor.New(tensor.WithShape(t.Shape()...), tensor.Of(tensor.Float32))
	iterator := t.Iterator()
	for i, err := iterator.Next(); err == nil; i, err = iterator.Next() {
		v := t.Get(i)

		// TODO: must be a more effiecient way.
		switch a := v.(type) {
		case float32:
			return t, nil
		case int:
			f := float32(a)
			new.Set(i, f)
		case int32:
			f := float32(a)
			new.Set(i, f)
		case int64:
			f := float32(a)
			new.Set(i, f)
		case float64:
			f := float32(a)
			new.Set(i, f)
		default:
			return nil, fmt.Errorf("could not cast type: %v", t.Dtype())
		}

	}
	return new, nil
}
