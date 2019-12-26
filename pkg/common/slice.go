package common

// MakeIntRange creates an int slice.
func MakeIntRange(min, max int) []int {
	a := make([]int, max-min+1)
	for i := range a {
		a[i] = min + i
	}
	return a
}

// MakeIRange create a
func MakeIRange(min, max int) []interface{} {
	a := make([]interface{}, max-min+1)
	for i := range a {
		a[i] = min + i
	}
	return a
}
