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
	Compile(x, y *Input, opts ...Opt) error

	// Predict x.
	Predict(x g.Value) (prediction g.Value, err error)

	// Fit x to y.
	Fit(x, y g.Value) error

	// FitBatch fits x to y as batches.
	FitBatch(x, y g.Value) error

	// PredictBatch predicts x as a batch
	PredictBatch(x g.Value) (prediction g.Value, err error)

	// Visualize the model by graph name.
	Visualize(name string)

	// Graph returns the expression graph for the model.
	Graphs() map[string]*g.ExprGraph

	// X is the input to the model.
	X() *Input

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

	x, y             *Input
	additionalInputs Inputs

	trainChain       *layers.Chain
	trainBatchChain  *layers.Chain
	onlineChain      *layers.Chain
	onlineBatchChain *layers.Chain

	trainGraph       *g.ExprGraph
	trainBatchGraph  *g.ExprGraph
	onlineGraph      *g.ExprGraph
	onlineBatchGraph *g.ExprGraph

	xTrain       *Input
	xTrainBatch  *Input
	xOnline      *Input
	xOnlineBatch *Input

	yTrain      *Input
	yTrainBatch *Input

	trainPredVal       g.Value
	trainBatchPredVal  g.Value
	onlinePredVal      g.Value
	onlineBatchPredVal g.Value

	batchSize int
	lossFn    LossFn
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
func WithLoss(lossFn LossFn) func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.lossFn = lossFn
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

// WithInputs adds an input to the graph for training.
func WithInputs(inputs ...*Input) func(Model) {
	return func(m Model) {
		switch t := m.(type) {
		case *Sequential:
			t.additionalInputs = append(t.additionalInputs, inputs...)
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

// Compile the model.
func (s *Sequential) Compile(x, y *Input, opts ...Opt) error {
	fmt.Println("compiling")
	x.Normalize()
	s.x = x

	y.Normalize()
	s.y = y
	fmt.Println("done compiling")

	for _, opt := range opts {
		opt(s)
	}
	if s.lossFn == nil {
		s.lossFn = MeanSquaredError
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
	err := s.buildTrainGraph(x, y)
	if err != nil {
		return err
	}
	err = s.buildTrainBatchGraph(x, y)
	if err != nil {
		return err
	}
	err = s.buildOnlineGraph(x)
	if err != nil {
		return err
	}
	err = s.buildOnlineBatchGraph(x)
	if err != nil {
		return err
	}
	return nil
}

func (s *Sequential) buildTrainGraph(x, y *Input) error {
	s.trainGraph = g.NewGraph()

	s.xTrain = x.Clone()
	s.xTrain.Compile(s.trainGraph)

	s.yTrain = y.Clone()
	s.yTrain.Compile(s.trainGraph)

	s.additionalInputs.Compile(s.trainGraph)

	s.trainChain = s.Chain.Clone()
	s.trainChain.Compile(s.xTrain.Node())

	prediction, err := s.trainChain.Fwd(s.xTrain.Node())
	if err != nil {
		return err
	}
	g.Read(prediction, &s.trainPredVal)

	loss, err := s.lossFn(prediction, s.yTrain.Node())
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

func (s *Sequential) buildTrainBatchGraph(x, y *Input) error {
	s.trainBatchGraph = g.NewGraph()

	s.xTrainBatch = s.x.AsBatch(s.batchSize)
	s.xTrainBatch.Compile(s.trainBatchGraph)

	s.yTrainBatch = s.y.AsBatch(s.batchSize)
	s.yTrainBatch.Compile(s.trainBatchGraph)

	s.additionalInputs.Clone().Compile(s.trainBatchGraph)

	s.trainBatchChain = s.Chain.Clone()
	s.trainBatchChain.Compile(s.xTrainBatch.Node(), layers.WithSharedChainLearnables(s.trainChain), layers.WithLayerOpts(layers.AsBatch()))

	prediction, err := s.trainBatchChain.Fwd(s.xTrainBatch.Node())
	if err != nil {
		return err
	}
	g.Read(prediction, &s.trainBatchPredVal)

	loss, err := s.lossFn(prediction, s.yTrainBatch.Node())
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

func (s *Sequential) buildOnlineGraph(x *Input) error {
	s.onlineGraph = g.NewGraph()

	s.xOnline = s.x.Clone()
	s.xOnline.Compile(s.onlineGraph)

	s.onlineChain = s.Chain.Clone()
	s.onlineChain.Compile(s.xOnline.Node(), layers.WithSharedChainLearnables(s.trainChain))

	prediction, err := s.onlineChain.Fwd(s.xOnline.Node())
	if err != nil {
		return err
	}
	g.Read(prediction, &s.onlinePredVal)
	s.onlineVM = g.NewTapeMachine(s.onlineGraph)
	return nil
}

func (s *Sequential) buildOnlineBatchGraph(x *Input) error {
	s.onlineBatchGraph = g.NewGraph()

	s.xOnlineBatch = s.x.AsBatch(s.batchSize)
	s.xOnlineBatch.Compile(s.onlineBatchGraph)

	s.onlineBatchChain = s.Chain.Clone()
	s.onlineBatchChain.Compile(s.xOnlineBatch.Node(), layers.WithSharedChainLearnables(s.trainChain), layers.WithLayerOpts(layers.AsBatch()))

	prediction, err := s.onlineBatchChain.Fwd(s.xOnlineBatch.Node())
	if err != nil {
		return err
	}
	g.Read(prediction, &s.onlineBatchPredVal)
	s.onlineBatchVM = g.NewTapeMachine(s.onlineBatchGraph)
	return nil
}

// Predict x.
func (s *Sequential) Predict(x g.Value) (prediction g.Value, err error) {
	err = s.xOnline.Set(x)
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
	if !x.Shape().Eq(s.xOnlineBatch.Shape()) {
		return nil, fmt.Errorf("attempting to run predict batch with the wrong shape %v, predict batch expects %v", x.Shape(), s.xOnlineBatch.Shape())
	}
	err = s.xOnlineBatch.Set(x)
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
func (s *Sequential) Fit(x, y g.Value) error {
	if !x.Shape().Eq(s.xTrain.Shape()) {
		return fmt.Errorf("attempting to run fit with the wrong x shape %v, fit expects %v", x.Shape(), s.xTrain.Shape())
	}
	if !y.Shape().Eq(s.yTrain.Shape()) {
		return fmt.Errorf("attempting to run fit with the wrong y shape %v, fit expects %v", y.Shape(), s.yTrain.Shape())
	}
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
func (s *Sequential) FitBatch(x, y g.Value) error {
	if !x.Shape().Eq(s.xTrainBatch.Shape()) {
		return fmt.Errorf("attempting to run fit batch with the wrong x batch shape %v, fit batch expects %v", x.Shape(), s.xTrainBatch.Shape())
	}
	if !y.Shape().Eq(s.yTrainBatch.Shape()) {
		return fmt.Errorf("attempting to run fit batch with the wrong y batch shape %v, fit batch expects %v", y.Shape(), s.yTrainBatch.Shape())
	}
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
func (s *Sequential) X() *Input {
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
