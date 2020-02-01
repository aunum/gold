package model

import (
	"fmt"

	"github.com/pbarker/log"

	g "gorgonia.org/gorgonia"
	t "gorgonia.org/tensor"
)

// Input into the model.
type Input struct {
	name  string
	shape t.Shape
	dtype t.Dtype
	node  *g.Node
}

// InputOpt is an input option.
type InputOpt func(*Input)

// AsType explicitly sets the type of the input.
// Defaults to Float32.
func AsType(dtype t.Dtype) func(*Input) {
	return func(i *Input) {
		i.dtype = dtype
	}
}

// NewInput returns a new input.
func NewInput(name string, shape t.Shape, opts ...InputOpt) *Input {
	i := &Input{
		name:  name,
		shape: shape,
		dtype: t.Float32,
	}
	for _, opt := range opts {
		opt(i)
	}
	return i
}

// Compile an input into a graph.
func (i *Input) Compile(graph *g.ExprGraph, opts ...InputOpt) *g.Node {
	for _, opt := range opts {
		opt(i)
	}
	n := g.NewTensor(graph, i.dtype, len(i.shape), g.WithShape(i.shape...), g.WithName(i.name))
	i.node = n
	return n
}

// Shape is the shape of the input.
func (i *Input) Shape() t.Shape {
	return i.shape
}

// Name of the input.
func (i *Input) Name() string {
	return i.name
}

// DType data type of the input.
func (i *Input) DType() t.Dtype {
	return i.dtype
}

// Node returns the graph node.
func (i *Input) Node() *g.Node {
	return i.node
}

// Clone an input.
func (i *Input) Clone() *Input {
	return &Input{
		name:  i.name,
		shape: i.shape.Clone(),
		dtype: i.dtype,
	}
}

// Check that the dimensions and type of the given value are congruent with the
// expected input.
func (i *Input) Check(value g.Value) error {
	vShape := value.Shape()
	if len(vShape) != len(i.Shape()) {
		return fmt.Errorf("shape mismatch: input %v expects %v got %v", i.name, i.shape, vShape)
	}
	for index, s := range i.Shape() {
		if vShape[index] != s {
			return fmt.Errorf("shape mismatch: input %v expects %v got %v", i.name, i.shape, vShape)
		}
	}
	if i.dtype != value.Dtype() {
		return fmt.Errorf("data type mismatch: input %v expects %v got %v", i.name, i.dtype, value.Dtype())
	}
	return nil
}

// Set the value of the input.
func (i *Input) Set(value g.Value) error {
	err := i.Check(value)
	if err != nil {
		return err
	}
	return g.Let(i.node, value)
}

// AsBatch converts an input to a batched representation.
func (i *Input) AsBatch(size int) *Input {
	ret := i.Clone()
	ret.shape[0] = size
	ret.name = fmt.Sprintf("%v_batch", ret.name)
	return ret
}

// Normalize the input. If the input is a scalar it will expand it to a matrix.
func (i *Input) Normalize() {
	if len(i.shape) == 1 {
		i.shape = []int{1, i.shape[0]}
		log.Infof("normalizing dimensions of %q to %v", i.name, i.shape)
	}
	if i.shape[0] != 1 {
		log.Fatalf("input shape %q %v must be a scalar or have the first dimension 1 e.g. [1, 4]", i.name, i.shape)
	}
}

// Inputs is a slice of input.
type Inputs []*Input

// Get an input by name.
func (i Inputs) Get(name string) (*Input, error) {
	for _, input := range i {
		if input.name == name {
			return input, nil
		}
	}
	return nil, fmt.Errorf("could not find input %s", name)
}

// Compile all inputs into the given graph.
func (i Inputs) Compile(graph *g.ExprGraph, opts ...InputOpt) g.Nodes {
	nodes := g.Nodes{}
	for _, input := range i {
		n := input.Compile(graph, opts...)
		nodes = append(nodes, n)
	}
	return nodes
}

// Clone the inputs.
func (i Inputs) Clone() Inputs {
	inputs := Inputs{}
	for _, input := range i {
		inputs = append(inputs, input.Clone())
	}
	return inputs
}
