package space

// Space is a space in which an agent can act.
type Space interface {
	// N is the number of possible states within a space.
	N() int

	// Sample of the space.
	Sample() interface{}

	// Whether the space contains the given value.
	Contains(v interface{}) bool
}
