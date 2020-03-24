package main

import (
	"github.com/aunum/gold/pkg/v1/agent/nes"
	"github.com/aunum/gold/pkg/v1/common/require"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	"github.com/aunum/log"
)

func main() {
	errorLogger := log.NewLogger(log.ErrorLevel, true)

	serverConfig := envv1.GymServerConfig
	serverConfig.Logger = errorLogger
	server, err := envv1.NewLocalServer(envv1.GymServerConfig)
	require.NoError(err)
	defer server.Close()

	blackBoxConfig := nes.DefaultSphereBlackBoxConfig
	blackBoxConfig.Logger = errorLogger
	blackBox, err := nes.NewSphereBlackBox(blackBoxConfig, server)
	require.NoError(err)

	hypers := &nes.EvolverHyperparameters{
		NPop:  50,
		NGen:  300,
		Sigma: 0.01,
		Alpha: 0.001,
	}
	config := &nes.EvolverConfig{
		EvolverHyperparameters: hypers,
		BlackBox:               blackBox,
	}
	evolver := nes.NewEvolver(config)
	require.NoError(err)

	finalWeights, err := evolver.Evolve()
	require.NoError(err)

	log.Debugvb("final weights", finalWeights)
}
