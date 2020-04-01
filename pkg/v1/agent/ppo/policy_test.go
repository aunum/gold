package ppo_test

import (
	"testing"

	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	. "github.com/aunum/gold/pkg/v1/agent/ppo"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	sphere "github.com/aunum/gold/pkg/v1/env"

	"github.com/aunum/log"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestPolicy(t *testing.T) {

	// test that network converges to static values.
	s, err := sphere.NewLocalServer(sphere.GymServerConfig)
	require.NoError(t, err)
	defer s.Close()

	env, err := s.Make("CartPole-v1")
	require.NoError(t, err)

	base := agentv1.NewBase("test")
	m, err := MakeCritic(DefaultCriticConfig, base, env)
	require.NoError(t, err)

	xShape1 := env.ObservationSpaceShape()[0]
	x1 := tensor.New(tensor.WithShape(1, xShape1), tensor.WithBacking([]float32{0.051960364, 0.14512223, 0.12799974, -2.0140305}))

	xShape2 := env.ObservationSpaceShape()[0]
	x2 := tensor.New(tensor.WithShape(1, xShape2), tensor.WithBacking([]float32{0.15163246, -0.94560495, 0.3904586, -8.20394809}))

	yShape1 := envv1.PotentialsShape(env.ActionSpace)[0]
	y1 := tensor.New(tensor.WithShape(1, yShape1), tensor.WithBacking([]float32{0.4484117, 0.09160687}))

	yShape2 := envv1.PotentialsShape(env.ActionSpace)[0]
	y2 := tensor.New(tensor.WithShape(1, yShape2), tensor.WithBacking([]float32{3.23904234, 2.203948023}))

	qv1, err := m.Predict(x1)
	require.NoError(t, err)
	log.Infov("initial prediction x1", qv1)

	qv2, err := m.Predict(x2)
	require.NoError(t, err)
	log.Infov("initial prediction x2", qv2)

	err = m.Fit(x1, y1)
	require.NoError(t, err)

	err = m.Fit(x2, y2)
	require.NoError(t, err)

	base.Serve()
	for i := 0; i < 10000; i++ {
		j := float32(i)
		state := tensor.New(tensor.WithShape(1, 4), tensor.WithBacking([]float32{j * .001, j * -.002, j * .003, j * -.004}))
		log.Info(state)
		pred, err := m.Predict(state)
		log.Info(pred)
		require.NoError(t, err)
		err = m.Fit(x1, y1)
		require.NoError(t, err)
		err = m.Fit(x2, y2)
		require.NoError(t, err)
		base.Tracker.LogStep(i, 0)
	}
	qvf1, err := m.Predict(x1)
	require.NoError(t, err)

	qvf2, err := m.Predict(x2)
	require.NoError(t, err)
	log.Info("----")
	log.Infov("initial prediction x1", qv1)
	log.Infov("expected 1", y1)
	log.Infov("final prediction 1", qvf1)
	log.Info("----")
	log.Infov("initial prediction x2", qv2)
	log.Infov("expected 2", y2)
	log.Infov("final prediction 2", qvf2)
}
