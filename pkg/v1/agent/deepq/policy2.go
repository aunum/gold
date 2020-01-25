package deepq

// import (
// 	"fmt"

// 	agentv1 "github.com/pbarker/go-rl/pkg/v1/agent"
// 	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
// 	modelv1 "github.com/pbarker/go-rl/pkg/v1/model"
// 	"github.com/pbarker/go-rl/pkg/v1/model/layers"
// 	. "gorgonia.org/golgi"
// 	g "gorgonia.org/gorgonia"
// 	"gorgonia.org/tensor"
// )

// // PolicyConfig2 are the hyperparameters for a policy.
// type PolicyConfig2 struct {
// 	// Loss function to evaluate network perfomance.
// 	LossFn modelv1.LossFn

// 	// Optimizer to optimize the wieghts with regards to the error.
// 	Optimizer g.Solver

// 	// LayerBuilder is a builder of layers.
// 	LayerBuilder LayerBuilder
// }

// // DefaultPolicyConfig are the default hyperparameters for a policy.
// var DefaultPolicyConfig2 = &PolicyConfig{
// 	LossFn:       modelv1.MeanSquaredError,
// 	Optimizer:    g.NewAdamSolver(g.WithLearnRate(0.001)),
// 	LayerBuilder: DefaultFCLayerBuilder,
// }

// // LayerBuilder2 builds layers.
// type LayerBuilder2 func(env *envv1.Env) []modelv1.Layer

// // DefaultFCLayerBuilder2 is a default fully connected layer builder.
// var DefaultFCLayerBuilder2 = func(env *envv1.Env) []modelv1.Layer {
// 	return []modelv1.Layer{
// 		layers.NewFC(env.ObservationSpaceShape()[0], 100, layers.WithActivation(g.Rectify)),
// 		layers.NewFC(100, 100, layers.WithActivation(g.Rectify)),
// 		layers.NewFC(100, envv1.PotentialsShape(env.ActionSpace)[0]),
// 	}
// }

// // MakePolicy2 makes a model.
// func MakePolicy2(name string, config *PolicyConfig, base *agentv1.Base, env *envv1.Env) (modelv1.Model, error) {
// 	x := tensor.Ones(tensor.Float32, env.ObservationSpaceShape()[0])
// 	y := tensor.Ones(tensor.Float32, envv1.PotentialsShape(env.ActionSpace)[0])

// 	model, err := modelv1.NewSequential(name, x, y)
// 	if err != nil {
// 		return nil, err
// 	}
// 	model.AddLayers(config.LayerBuilder(env)...)

// 	err = model.Compile(
// 		modelv1.WithOptimizer(config.Optimizer),
// 		modelv1.WithLoss(config.LossFn),
// 		modelv1.WithTracker(base.Tracker),
// 	)
// 	if err != nil {
// 		return nil, err
// 	}
// 	return model, nil
// }

// func Example() {
// 	n := 100
// 	of := tensor.Float64
// 	g := g.NewGraph()
// 	x := g.NewTensor(g, of, 4, g.WithName("X"), g.WithShape(n, 1, 28, 28), g.WithInit(g.GlorotU(1)))
// 	y := g.NewMatrix(g, of, g.WithName("Y"), g.WithShape(n, 10), g.WithInit(g.GlorotU(1)))
// 	nn, err := ComposeSeq(
// 		x,
// 		L(ConsReshape, ToShape(n, 784)),
// 		L(ConsFC, WithSize(50), WithName("l0"), AsBatched(true), WithActivation(g.Tanh), WithBias(true)),
// 		L(ConsDropout, WithProbability(0.5)),
// 		L(ConsFC, WithSize(150), WithName("l1"), AsBatched(true), WithActivation(g.Rectify)), // by default WithBias is true
// 		L(ConsLayerNorm, WithSize(20), WithName("Norm"), WithEps(0.001)),
// 		L(ConsFC, WithSize(10), WithName("l2"), AsBatched(true), WithActivation(softmax), WithBias(false)),
// 	)
// 	if err != nil {
// 		panic(err)
// 	}
// 	out := nn.Fwd(x)
// 	if err = g.CheckOne(out); err != nil {
// 		panic(err)
// 	}

// 	cost := g.Must(RMS(out, y))
// 	model := nn.Model()
// 	if _, err = g.Grad(cost, model...); err != nil {
// 		panic(err)
// 	}
// 	m := g.NewTapeMachine(g)
// 	if err := m.RunAll(); err != nil {
// 		panic(err)
// 	}

// 	fmt.Printf("Model: %v\n", model)
// 	// Output:
// 	// Model: [l0_W, l0_B, l1_W, l1_B, Norm_W, Norm_B, l2_W]
// }
