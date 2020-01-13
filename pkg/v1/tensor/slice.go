package tensor

// RS is a ranged slice that implements the Slice interface.
type RS struct {
	start, end, step int
}

// Start of the slice.
func (s RS) Start() int { return s.start }

// End of the slice.
func (s RS) End() int { return s.end }

// Step of slice.
func (s RS) Step() int { return s.step }

// MakeRS creates a ranged slice. It takes an optional step param.
func MakeRS(start, end int, opts ...int) RS {
	step := 1
	if len(opts) > 0 {
		step = opts[0]
	}
	return RS{
		start: start,
		end:   end,
		step:  step,
	}
}

// SS is a single slice, representing this: [start:start+1:0]
type SS int

// Start of slice.
func (s SS) Start() int { return int(s) }

// End of the slice.
func (s SS) End() int { return int(s) + 1 }

// Step of slice.
func (s SS) Step() int { return 0 }
