package tensor

// // CartesianProduct returns the cartesian product of the given vector internals.
// func CartesianProduct(intervals ...*tensor.Dense) (*tensor.Dense, error) {
// 	// TODO: check all intervals
// 	final := tensor.New(tensor.WithShape(1, len(intervals)), tensor.Of(intervals[0].Dtype()))
// 	for _, interval := range intervals {
// 		if !interval.IsVector() {
// 			return nil, fmt.Errorf("all intervals must be vectors; %v is not a vector", interval)
// 		}
// 		iterator := interval.Iterator()
// 		for i, err := iterator.Next(); err == nil; i, err = iterator.Next() {
// 			fmt.Println("i: ", i)
// 		}
// 	}
// 	return nil, nil
// }

// func cartesianProduct(a, b *tensor.Dense) (*tensor.Dense, error) {
// 	iterA := a.Iterator()
// 	for i, err := iterA.Next(); err == nil; i, err = iterA.Next() {
// 		iterB := b.Iterator()
// 		fmt.Println("i: ", i)
// 		for j, err := iterB.Next(); err == nil; j, err = iterB.Next() {
// 			fmt.Println("j: ", j)
// 			return nil, nil
// 		}
// 	}
// 	return nil, nil
// }
