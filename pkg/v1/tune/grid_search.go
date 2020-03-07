package tune

import (
	"fmt"
	"sort"
	"sync"
	"time"

	. "github.com/pbarker/go-rl/pkg/v1/agent/q"
	. "github.com/pbarker/go-rl/pkg/v1/agent/q/experiments/envs"
	"github.com/pbarker/go-rl/pkg/v1/common/require"
	sphere "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/pbarker/log"
	"github.com/schwarmco/go-cartesian-product"
)

// TODO: make generic
func GridSearch() {
	s, err := sphere.NewLocalServer(sphere.GymServerConfig)
	require.NoError(err)
	defer s.Resource.Close()

	alpha := []interface{}{0.1, 0.2, 0.3, 0.5}
	epsilon := []interface{}{0.0, 0.01, 0.1, 0.2}
	gamma := []interface{}{1.0, 0.9, .7, .5}
	ada := []interface{}{5.0, 25.0, 50.0, 100.0}
	buckets := []interface{}{[]int{1, 1, 6, 3}, []int{1, 1, 3, 6}, []int{1, 1, 6, 12}, []int{1, 1, 12, 6}, []int{1, 1, 4, 4}}

	c := cartesian.Iter(alpha, epsilon, gamma, ada, buckets)

	configs := []CartPoleTestConfig{}
	for params := range c {
		conf := CartPoleTestConfig{
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
		config  CartPoleTestConfig
	}
	ch := make(chan *result)
	var wg sync.WaitGroup
	for i, conf := range configs {
		wg.Add(1)
		go func(s *sphere.Server, c CartPoleTestConfig, res chan *result) {
			defer wg.Done()
			r, err := TestCartPole(s, c)
			if err != nil {
				log.Error(err)
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
	results := map[float32]CartPoleTestConfig{}
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
