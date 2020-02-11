package space

import (
	"math/rand"
	"time"

	"github.com/pbarker/go-rl/pkg/v1/common/num"
	"gorgonia.org/tensor"
)

// Discrete space is an unsigned vector of whole numbers.
type Discrete struct {
	n      int
	values []int
	r      *rand.Rand
}

// NewDiscrete returns a new Discrete space.
func NewDiscrete(n int) *Discrete {
	s := rand.NewSource(time.Now().Unix())
	r := rand.New(s)
	return &Discrete{
		n:      n,
		values: num.MakeIRange(0, n),
		r:      r,
	}
}

// N is the number of discrete states.
func (d *Discrete) N() int {
	return d.n
}

// Sample from the discrete space.
func (d *Discrete) Sample() interface{} {
	return d.r.Intn(d.n)
}

// Contains checks whether a space contains the given value.
func (d *Discrete) Contains(v interface{}) bool {
	i, ok := v.(int)
	if !ok {
		return false
	}
	for _, val := range d.values {
		if i == val {
			return true
		}
	}
	return false
}

// Values returns the values of the discrete space.
func (d *Discrete) Values() []int {
	return d.values
}

// Dense returns the values of the discrete space as a tensor.
func (d *Discrete) Dense() *tensor.Dense {
	return tensor.New(tensor.WithShape(1, d.n), tensor.WithBacking(tensor.Range(tensor.Int, 0, d.n)))
}
