package dense

import t "gorgonia.org/tensor"

// Div safely divides 'a' by 'b' by checking if b is zero and slightly augmenting it if so.
//
// TODO: check efficiency, may be cheaper to just always add a small value.
func Div(a, b *t.Dense) (*t.Dense, error) {
	err := NormalizeZeros(b)
	if err != nil {
		return nil, err
	}
	return a.Div(b)
}
