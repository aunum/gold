package q_test

import (
	"fmt"
	"sort"
	"sync"
	"testing"
	"time"

	. "github.com/pbarker/go-rl/pkg/agent/q"
	"github.com/pbarker/go-rl/pkg/common"
	"github.com/pbarker/logger"
	sphere "github.com/pbarker/sphere/pkg/env"
	"github.com/schwarmco/go-cartesian-product"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestAgent(t *testing.T) {
	s, err := sphere.NewLocalServer(sphere.GymServerConfig)
	require.Nil(t, err)
	defer s.Resource.Close()

	_, err = testCartPole(s, cartPoleConfig{
		Hyperparameters: DefaultHyperparameters,
		Buckets:         []int{1, 1, 6, 12},
		NumEpisodes:     30,
	})
	require.Nil(t, err)
}

type cartPoleConfig struct {
	*Hyperparameters
	Buckets     []int
	NumEpisodes int
}

func testCartPole(s *sphere.Server, c cartPoleConfig) (*sphere.Results, error) {
	logger.Infoy("config", c)
	env, err := s.Make("CartPole-v0")
	if err != nil {
		return nil, err
	}

	// Get the size of the action space.
	actionSpaceSize := env.GetActionSpace().GetDiscrete().GetN()

	// Create the Q-learning agent.
	agent := NewAgent(c.Hyperparameters, int(actionSpaceSize), nil)

	// fmt.Printf("space: %+v\n", env.GetObservationSpace().GetBox())

	// discretize the box space.
	box, err := env.BoxSpace()
	if err != nil {
		return nil, err
	}
	box = reframeBox(box)
	// fmt.Printf("box space: %v \n", box)
	// per https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
	intervals := tensor.New(tensor.WithShape(box.Shape...), tensor.WithBacking(c.Buckets))
	obvBinner, err := common.NewDenseEqWidthBinner(intervals, box.Low, box.High)
	if err != nil {
		return nil, err
	}
	// fmt.Printf("binner: %#v\n", obvBinner)
	// fmt.Printf("widths: %#v\n", obvBinner.Widths())
	// fmt.Printf("bounds: %#v\n", obvBinner.Bounds())

	logger.Infof("running for %d episodes", c.NumEpisodes)
	for i := 0; i <= c.NumEpisodes; i++ {
		state, err := env.Reset()
		if err != nil {
			return nil, err
		}
		discreteState, err := obvBinner.Bin(state)
		if err != nil {
			return nil, err
		}
		// fmt.Printf("state: %v\n", state)
		// fmt.Printf("discreteState: %v\n", discreteState)

		for ts := 0; ts <= int(env.MaxEpisodeSteps); ts++ {
			// Adapt hyperparams
			agent.Adapt(ts)

			// Get an action from the agent.
			action, err := agent.Action(discreteState)
			if err != nil {
				return nil, err
			}

			outcome, err := env.Step(action)
			if err != nil {
				return nil, err
			}
			discreteObv, err := obvBinner.Bin(outcome.Observation)
			// fmt.Printf("observation: %v\n", outcome.Observation)
			// fmt.Printf("dobservation: %v\n", discreteObv)

			// Learn!
			err = agent.Learn(action, outcome.Reward, discreteState, discreteObv)
			if err != nil {
				return nil, err
			}

			if outcome.Done {
				logger.Successf("Episode %d finished after %d timesteps", i, ts+1)
				agent.Visualize()
				// should we clear the state table here?
				break
			}
			discreteState = discreteObv
		}
	}
	res, err := env.Results()
	if err != nil {
		return nil, err
	}
	env.End()
	return res, nil
}

func TestGridSearch(t *testing.T) {
	s, err := sphere.NewLocalServer(sphere.GymServerConfig)
	require.Nil(t, err)
	defer s.Resource.Close()

	alpha := []interface{}{0.1, 0.2, 0.3, 0.5}
	epsilon := []interface{}{0.0, 0.01, 0.1, 0.2}
	gamma := []interface{}{1.0, 0.9, .7, .5}
	ada := []interface{}{5.0, 25.0, 50.0, 100.0}
	buckets := []interface{}{[]int{1, 1, 6, 3}, []int{1, 1, 3, 6}, []int{1, 1, 6, 12}, []int{1, 1, 12, 6}, []int{1, 1, 4, 4}}

	c := cartesian.Iter(alpha, epsilon, gamma, ada, buckets)

	configs := []cartPoleConfig{}
	for params := range c {
		conf := cartPoleConfig{
			Hyperparameters: &Hyperparameters{
				Alpha:      float32(params[0].(float64)),
				Epsilon:    float32(params[1].(float64)),
				Gamma:      float32(params[2].(float64)),
				AdaDivisor: float32(params[3].(float64)),
			},
			Buckets:     params[4].([]int),
			NumEpisodes: 30,
		}
		configs = append(configs, conf)
	}

	type result struct {
		results *sphere.Results
		config  cartPoleConfig
	}
	ch := make(chan *result)
	var wg sync.WaitGroup
	for i, conf := range configs {
		wg.Add(1)
		go func(s *sphere.Server, c cartPoleConfig, res chan *result) {
			defer wg.Done()
			r, err := testCartPole(s, c)
			if err != nil {
				logger.Error(err)
			}
			res <- &result{
				results: r,
				config:  conf,
			}
		}(s, conf, ch)
		// don't overload the local server.
		if i%4 == 0 {
			time.Sleep(15 * time.Second)
		}
	}
	wg.Wait()
	close(ch)
	results := map[float32]cartPoleConfig{}
	for res := range ch {
		results[res.results.AverageReward] = res.config
	}
	rewards := []float64{}
	for reward := range results {
		rewards = append(rewards, float64(reward))
	}
	sort.Float64s(rewards)
	for reward := range rewards {
		fmt.Printf("reward: %v config: %#v\n", reward, results[float32(reward)])
	}
}

// these bounds are currently set to infinite bounds, need to be reframed to be made discrete.
func reframeBox(box *sphere.BoxSpace) *sphere.BoxSpace {
	box.Low.Set(1, float32(-2.5))
	box.High.Set(1, float32(2.5))
	box.Low.Set(3, float32(-4.0))
	box.High.Set(3, float32(4.0))
	return box
}
