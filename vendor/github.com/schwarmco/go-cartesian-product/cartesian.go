package cartesian

import "sync"

// Iter takes interface-slices and returns a channel, receiving cartesian products
func Iter(params ...[]interface{}) chan []interface{} {
	// create channel
	c := make(chan []interface{})
	// create waitgroup
	var wg sync.WaitGroup
	// call iterator
	wg.Add(1)
	iterate(&wg, c, []interface{}{}, params...)
	// call channel-closing go-func
	go func() { wg.Wait(); close(c) }()
	// return channel
	return c
}

// private, recursive Iteration-Function
func iterate(wg *sync.WaitGroup, channel chan []interface{}, result []interface{}, params ...[]interface{}) {
	// dec WaitGroup when finished
	defer wg.Done()
	// no more params left?
	if len(params) == 0 {
		// send result to channel
		channel <- result
		return
	}
	// shift first param
	p, params := params[0], params[1:]
	// iterate over it
	for i := 0; i < len(p); i++ {
		// inc WaitGroup
		wg.Add(1)
		// create copy of result
		resultCopy := append([]interface{}{}, result...)
		// call self with remaining params
		go iterate(wg, channel, append(resultCopy, p[i]), params...)
	}
}
