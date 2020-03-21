package her_test

import (
	"testing"

	agentv1 "github.com/aunum/gold/pkg/v1/agent"
	. "github.com/aunum/gold/pkg/v1/agent/her"
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

	env, err := s.Make("BitFlipper-v0")
	require.NoError(t, err)

	base := agentv1.NewBase("test")
	m, err := MakePolicy("test", DefaultPolicyConfig, base, env)
	require.NoError(t, err)

	xShape1 := env.ObservationSpaceShape()[0] * 2
	x1 := tensor.New(tensor.WithShape(1, xShape1), tensor.WithBacking([]float32{0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0}))

	xShape2 := env.ObservationSpaceShape()[0] * 2
	x2 := tensor.New(tensor.WithShape(1, xShape2), tensor.WithBacking([]float32{1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0}))

	yShape1 := envv1.PotentialsShape(env.ActionSpace)[0]
	y1 := tensor.New(tensor.WithShape(1, yShape1), tensor.WithBacking(tensor.Range(tensor.Float32, 0, 11)))

	yShape2 := envv1.PotentialsShape(env.ActionSpace)[0]
	y2 := tensor.New(tensor.WithShape(1, yShape2), tensor.WithBacking(tensor.Range(tensor.Float32, 10, 21)))

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
		pred, err := m.Predict(x1)
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
	// time.Sleep(60 * time.Second)
}
