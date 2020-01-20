package deepq_test

import (
	"testing"

	. "github.com/pbarker/go-rl/pkg/v1/agent/deepq"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	sphere "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/pbarker/logger"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestPolicy(t *testing.T) {

	// test that network converges to static values.
	s, err := sphere.NewLocalServer(sphere.GymServerConfig)
	require.NoError(t, err)
	defer s.Resource.Close()

	env, err := s.Make("CartPole-v0")
	require.NoError(t, err)

	m, err := MakePolicy(DefaultPolicyConfig, env)
	require.NoError(t, err)

	xShape := env.ObservationSpaceShape()[0]
	x := tensor.New(tensor.WithShape(xShape), tensor.WithBacking([]float32{0.051960364, 0.14512223, 0.12799974, 0.63951147}))

	yShape := envv1.PotentialsShape(env.ActionSpace)[0]
	y := tensor.New(tensor.WithShape(yShape), tensor.WithBacking([]float32{0.4484117, 0.09160687}))

	qv1, err := m.Predict(x)
	require.NoError(t, err)
	logger.Infov("initial prediction", qv1)

	err = m.Fit(x, y)
	require.NoError(t, err)

	for i := 0; i < 1000; i++ {
		err = m.Fit(x, y)
		require.NoError(t, err)
	}
	qvf, err := m.Predict(x)
	require.NoError(t, err)
	logger.Infov("expected", y)
	logger.Infov("final prediction", qvf)
}
