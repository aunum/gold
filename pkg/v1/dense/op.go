package dense

import t "gorgonia.org/tensor"

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
