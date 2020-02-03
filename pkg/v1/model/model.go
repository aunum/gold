package model

import (
	"fmt"

	"github.com/pbarker/go-rl/pkg/v1/common"
	"github.com/pbarker/go-rl/pkg/v1/model/layers"
	"github.com/pbarker/go-rl/pkg/v1/track"
	"github.com/pbarker/log"
	g "gorgonia.org/gorgonia"
)

// Model is a prediction model.
type Model interface {
	// Compile the model.
	Compile(x In, y *Input, opts ...Opt) error

	// Predict x.
	Predict(x g.Value) (prediction g.Value, err error)

	// Fit x to y.
	Fit(x Values, y g.Value) error

	// FitBatch fits x to y as batches.
	FitBatch(x Values, y g.Value) error

	// PredictBatch predicts x as a batch
	PredictBatch(x g.Value) (prediction g.Value, err error)

	// Visualize the model by graph name.
	Visualize(name string)

	// Graph returns the expression graph for the model.
	Graphs() map[string]*g.ExprGraph

	// X is the inputs to the model.
	X() In

	// Y is the expected output of the model.
	Y() *Input

	// Learnables for the model.
	Learnables() g.Nodes
}

// Sequential model.
type Sequential struct {
	// Chain of layers in the model.
	Chain *layers.Chain

	// Tracker of values.
	Tracker   *track.Tracker
	noTracker bool

	name string

	x   Inputs
	y   *Input
	fwd *Input

	trainChain       *layers.Chain
	trainBatchChain  *layers.Chain
	onlineChain      *layers.Chain
	onlineBatchChain *layers.Chain

	trainGraph       *g.ExprGraph
	trainBatchGraph  *g.ExprGraph
	onlineGraph      *g.ExprGraph
	onlineBatchGraph *g.ExprGraph

	xTrain          Inputs
	xTrainFwd       *Input
	xTrainBatch     Inputs
	xTrainBatchFwd  *Input
	xOnline         Inputs
	xOnlineFwd      *Input
	xOnlineBatch    Inputs
	xOnlineBatchFwd *Input

	yTrain      *Input
	yTrainBatch *Input

	trainPredVal       g.Value
	trainBatchPredVal  g.Value
	onlinePredVal      g.Value
	onlineBatchPredVal g.Value

	loss           Loss
	trainLoss      Loss
	trainBatchLoss Loss

	batchSize int
	optimizer g.Solver

	trainVM       g.VM
	trainBatchVM  g.VM
	onlineVM      g.VM
	onlineBatchVM g.VM
}

// NewSequential returns a new sequential model.
func NewSequential(name string) (*Sequential, error) {
	return &Sequential{
		Chain:     layers.NewChain(),
		name:      name,
		batchSize: 32,
	}, nil
}

// Opt is a model option.
type Opt func(Model)

// WithLoss uses a specific loss function with the model.
// Defaults to MSE.
func WithLoss(loss Loss) func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.loss = loss
		default:
			log.Fatal("unknown model type")
		}
	}
}

// WithOptimizer uses a specific optimizer function.
// Defaults to Adam.
func WithOptimizer(optimizer g.Solver) func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.optimizer = optimizer
		default:
			log.Fatal("unknown model type")
		}
	}
}

// WithTracker adds a tracker to the model, if not provided one will be created.
func WithTracker(tracker *track.Tracker) func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.Tracker = tracker
		default:
			log.Fatal("unknown model type")
		}
	}
}

// WithNoTracker uses no tracking with the model.
func WithNoTracker() func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.noTracker = true
		default:
			log.Fatal("unknown model type")
		}
	}
}

// WithBatchSize sets the batch size for the model.
// Defaults to 32.
func WithBatchSize(size int) func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.batchSize = size
		default:
			log.Fatal("unknown model type")
		}
	}
}

// AddLayer adds a layer.
func (s *Sequential) AddLayer(layer layers.Layer) {
	s.Chain.Add(layer)
}

// AddLayers adds a number of layers.
func (s *Sequential) AddLayers(layers ...layers.Layer) {
	for _, layer := range layers {
		s.Chain.Add(layer)
	}
}

// Fwd tells the model which input should be sent through the layers.
// If not provided, the first input will be used.
func (s *Sequential) Fwd(x *Input) {
	s.fwd = x
}

// Compile the model.
func (s *Sequential) Compile(x In, y *Input, opts ...Opt) error {
	s.x = x.Inputs()

	err := y.Normalize()
	if err != nil {
		return err
	}
	s.y = y

	for _, opt := range opts {
		opt(s)
	}
	if s.loss == nil {
		s.loss = MSE
	}
	if s.optimizer == nil {
		s.optimizer = g.NewAdamSolver()
	}
	if s.Tracker == nil && !s.noTracker {
		tracker, err := track.NewTracker()
		if err != nil {
			return err
		}
		s.Tracker = tracker
	}
	if s.fwd == nil {
		s.fwd = x.Inputs()[0]
		log.Infof("setting foward for layers to input %q", s.fwd.Name())
	}
	err = s.buildTrainGraph(s.x, y)
	if err != nil {
		return err
	}
	err = s.buildTrainBatchGraph(s.x, y)
	if err != nil {
		return err
	}
	err = s.buildOnlineGraph(s.x)
	if err != nil {
		return err
	}
	err = s.buildOnlineBatchGraph(s.x)
	if err != nil {
		return err
	}
	return nil
}

func (s *Sequential) buildTrainGraph(x Inputs, y *Input) (err error) {
	s.trainGraph = g.NewGraph()

	s.xTrain = x.Clone()
	s.xTrain.Compile(s.trainGraph)

	s.xTrainFwd, err = s.xTrain.Get(s.fwd.Name())
	if err != nil {
		return err
	}

	s.trainLoss = s.loss.CloneTo(s.trainGraph)

	s.yTrain = y.Clone()
	s.yTrain.Compile(s.trainGraph)

	s.trainChain = s.Chain.Clone()
	s.trainChain.Compile(s.trainGraph)

	prediction, err := s.trainChain.Fwd(s.xTrainFwd.Node())
	if err != nil {
		return err
	}
	g.Read(prediction, &s.trainPredVal)

	loss, err := s.trainLoss.Compute(prediction, s.yTrain.Node())
	if err != nil {
		return err
	}
	if s.Tracker != nil {
		s.Tracker.TrackValue("train_loss", loss, track.WithNamespace(s.name))
	}

	_, err = g.Grad(loss, s.trainChain.Learnables()...)
	if err != nil {
		return err
	}

	s.trainVM = g.NewTapeMachine(s.trainGraph, g.BindDualValues(s.trainChain.Learnables()...))
	return nil
}

func (s *Sequential) buildTrainBatchGraph(x Inputs, y *Input) (err error) {
	s.trainBatchGraph = g.NewGraph()

	for _, input := range x {
		if input.Name() == s.fwd.Name() {
			s.xTrainBatchFwd = input.AsBatch(s.batchSize)
			s.xTrainBatchFwd.Compile(s.trainBatchGraph)
			s.xTrainBatch = append(s.xTrainBatch, s.xTrainBatchFwd)
			continue
		}
		i := input.CloneTo(s.trainBatchGraph)
		s.xTrainBatch = append(s.xTrainBatch, i)
	}

	s.yTrainBatch = s.y.AsBatch(s.batchSize)
	s.yTrainBatch.Compile(s.trainBatchGraph)

	s.trainBatchLoss = s.loss.CloneTo(s.trainBatchGraph)

	s.trainBatchChain = s.Chain.Clone()
	s.trainBatchChain.Compile(s.trainBatchGraph, layers.WithSharedChainLearnables(s.trainChain), layers.WithLayerOpts(layers.AsBatch()))

	prediction, err := s.trainBatchChain.Fwd(s.xTrainBatchFwd.Node())
	if err != nil {
		return err
	}
	g.Read(prediction, &s.trainBatchPredVal)

	loss, err := s.trainBatchLoss.Compute(prediction, s.yTrainBatch.Node())
	if err != nil {
		return err
	}
	if s.Tracker != nil {
		s.Tracker.TrackValue("batch_loss", loss, track.WithNamespace(s.name))
	}

	_, err = g.Grad(loss, s.trainBatchChain.Learnables()...)
	if err != nil {
		return err
	}
	s.trainBatchVM = g.NewTapeMachine(s.trainBatchGraph, g.BindDualValues(s.trainBatchChain.Learnables()...))
	return nil
}

func (s *Sequential) buildOnlineGraph(x Inputs) (err error) {
	s.onlineGraph = g.NewGraph()

	s.xOnline = s.x.Clone()
	s.xOnline.Compile(s.onlineGraph)

	s.xOnlineFwd, err = s.xOnline.Get(s.fwd.Name())
	if err != nil {
		return err
	}

	s.onlineChain = s.Chain.Clone()
	s.onlineChain.Compile(s.onlineGraph, layers.WithSharedChainLearnables(s.trainChain))

	prediction, err := s.onlineChain.Fwd(s.xOnlineFwd.Node())
	if err != nil {
		return err
	}
	g.Read(prediction, &s.onlinePredVal)
	s.onlineVM = g.NewTapeMachine(s.onlineGraph)
	return nil
}

func (s *Sequential) buildOnlineBatchGraph(x Inputs) error {
	s.onlineBatchGraph = g.NewGraph()

	for _, input := range x {
		if input.Name() == s.fwd.Name() {
			s.xOnlineBatchFwd = input.AsBatch(s.batchSize)
			s.xOnlineBatchFwd.Compile(s.onlineBatchGraph)
			s.xOnlineBatch = append(s.xOnlineBatch, s.xOnlineBatchFwd)
			continue
		}
		i := input.CloneTo(s.onlineBatchGraph)
		s.xOnlineBatch = append(s.xOnlineBatch, i)
	}

	s.onlineBatchChain = s.Chain.Clone()
	s.onlineBatchChain.Compile(s.onlineBatchGraph, layers.WithSharedChainLearnables(s.trainChain), layers.WithLayerOpts(layers.AsBatch()))

	prediction, err := s.onlineBatchChain.Fwd(s.xOnlineBatchFwd.Node())
	if err != nil {
		return err
	}
	g.Read(prediction, &s.onlineBatchPredVal)
	s.onlineBatchVM = g.NewTapeMachine(s.onlineBatchGraph)
	return nil
}

// Predict x.
func (s *Sequential) Predict(x g.Value) (prediction g.Value, err error) {
	err = s.xOnlineFwd.Set(x)
	if err != nil {
		return prediction, err
	}
	err = s.onlineVM.RunAll()
	if err != nil {
		return prediction, err
	}
	prediction = s.onlinePredVal
	s.onlineVM.Reset()
	return
}

// PredictBatch predicts x as a batch.
func (s *Sequential) PredictBatch(x g.Value) (prediction g.Value, err error) {
	err = s.xOnlineBatchFwd.Set(x)
	if err != nil {
		return prediction, err
	}
	err = s.onlineBatchVM.RunAll()
	if err != nil {
		return prediction, err
	}
	prediction = s.onlineBatchPredVal
	s.onlineBatchVM.Reset()
	return
}

// Fit x to y.
func (s *Sequential) Fit(x Values, y g.Value) error {
	s.yTrain.Set(y)
	s.xTrain.Set(x)

	err := s.trainVM.RunAll()
	if err != nil {
		return err
	}
	grads := g.NodesToValueGrads(s.trainChain.Learnables())
	s.optimizer.Step(grads)
	s.trainVM.Reset()
	return nil
}

// FitBatch fits x to y as a batch.
func (s *Sequential) FitBatch(x Values, y g.Value) error {
	s.yTrainBatch.Set(y)
	s.xTrainBatch.Set(x)

	err := s.trainBatchVM.RunAll()
	if err != nil {
		return err
	}
	grads := g.NodesToValueGrads(s.trainBatchChain.Learnables())
	s.optimizer.Step(grads)
	s.trainBatchVM.Reset()
	return nil
}

// Visualize the model by graph name.
func (s *Sequential) Visualize(name string) {
	common.Visualize(s.Graphs()[name])
}

// Graphs returns the expression graphs for the model.
func (s *Sequential) Graphs() map[string]*g.ExprGraph {
	return map[string]*g.ExprGraph{
		"train":       s.trainGraph,
		"trainBatch":  s.trainBatchGraph,
		"online":      s.onlineGraph,
		"onlineBatch": s.onlineBatchGraph,
	}
}

// X is is the input to the model.
func (s *Sequential) X() In {
	return s.x
}

// Y is is the output of the model.
func (s *Sequential) Y() *Input {
	return s.y
}

// Learnables are the model learnables.
func (s *Sequential) Learnables() g.Nodes {
	return s.trainChain.Learnables()
}

// CloneLearnablesTo another model.
func (s *Sequential) CloneLearnablesTo(to *Sequential) error {
	desired := s.trainChain.Learnables()
	destination := to.trainChain.Learnables()
	if len(desired) != len(destination) {
		return fmt.Errorf("models must be identical to clone learnables")
	}
	for i, learnable := range destination {
		c := desired[i].Clone()
		err := g.Let(learnable, c.(*g.Node).Value())
		if err != nil {
			return err
		}
	}
	new := to.trainChain.Learnables()
	shared := map[string]*layers.Chain{
		"trainBatch":  to.trainBatchChain,
		"online":      to.onlineChain,
		"onlineBatch": to.onlineBatchChain,
	}
	for name, chain := range shared {
		log.Info("chain: ", name)
		for i, learnable := range chain.Learnables() {
			err := g.Let(learnable, new[i].Value())
			if err != nil {
				return err
			}
			log.Infovb(learnable.Name(), learnable.Value())
		}
	}
	return nil
}

// Opts are optsion for a model
type Opts struct {
	opts []Opt
}

// NewOpts returns a new set of options for a model.
func NewOpts() *Opts {
	return &Opts{opts: []Opt{}}
}

// Add an option to the options.
func (o *Opts) Add(opts ...Opt) {
	o.opts = append(o.opts, opts...)
}

// Values are the options.
func (o *Opts) Values() []Opt {
	return o.opts
}
