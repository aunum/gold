package env

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/aunum/gold/pkg/v1/common"
	"github.com/aunum/log"
	spherev1alpha "github.com/aunum/sphere/api/gen/go/v1alpha"
	"github.com/ory/dockertest"
	dc "github.com/ory/dockertest/docker"
	"google.golang.org/grpc"
)

// Server of environments.
type Server struct {
	// Client to connect to the Sphere server.
	Client spherev1alpha.EnvironmentAPIClient

	resource    *dockertest.Resource
	containerID string
	pool        *dockertest.Pool
	conn        *grpc.ClientConn
	logger      *log.Logger
	dialOpts    []grpc.DialOption
}

// LocalServerConfig is the environment server config.
type LocalServerConfig struct {
	// Docker image of environment.
	Image string

	// Version of the docker image.
	Version string

	// Port the environment is exposed on.
	Port string

	// Logger for the server.
	Logger *log.Logger
}

// GymServerConfig is a configuration for a OpenAI Gym server environment.
var GymServerConfig = &LocalServerConfig{Image: "sphereproject/gym", Version: "latest", Port: "50051/tcp"}

// NewLocalServer creates a new environment server by launching a docker container and connecting to it.
func NewLocalServer(config *LocalServerConfig) (*Server, error) {
	if config.Logger == nil {
		config.Logger = log.DefaultLogger
	}
	config.Logger.Info("creating local server")
	pool, err := dockertest.NewPool("")
	if err != nil {
		return nil, fmt.Errorf("Could not connect to docker: %s", err)
	}

	resource, err := pool.Run(config.Image, config.Version, []string{})
	if err != nil {
		return nil, fmt.Errorf("Could not start resource: %s", err)
	}

	addr := fmt.Sprintf("localhost:%s", resource.GetPort(config.Port))
	client, conn, err := connect(addr, config.Logger, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		return nil, err
	}

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigs
		config.Logger.Warningf("closing env on sig %s", sig.String())
		resource.Close()
		os.Exit(1)
	}()

	return &Server{
		Client:   client,
		conn:     conn,
		resource: resource,
		pool:     pool,
		logger:   config.Logger,
	}, nil
}

// FindOrCreate will find a local server for the given config or create one.
func FindOrCreate(config *LocalServerConfig) (*Server, error) {
	if config.Logger == nil {
		config.Logger = log.DefaultLogger
	}
	pool, err := dockertest.NewPool("")
	if err != nil {
		return nil, fmt.Errorf("Could not connect to docker: %s", err)
	}
	containers, err := pool.Client.ListContainers(dc.ListContainersOptions{
		All: true,
		Filters: map[string][]string{
			"ancestor": {config.Image},
		},
	})
	if err != nil {
		return nil, err
	}
	if len(containers) == 0 {
		config.Logger.Info("no existing containers found, creating new...")
		return NewLocalServer(config)
	}
	container := containers[0]
	config.Logger.Infof("connecting to existing container %q", strings.TrimPrefix(container.Names[0], "/"))
	port := container.Ports[0]

	addr := fmt.Sprintf("localhost:%d", port.PublicPort)
	client, conn, err := connect(addr, config.Logger, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		return nil, err
	}
	return &Server{
		Client:      client,
		conn:        conn,
		pool:        pool,
		containerID: container.ID,
		logger:      config.Logger,
	}, nil
}

// Connect to a server.
func Connect(addr string, opts ...ServerOpts) (*Server, error) {
	s := &Server{}
	for _, opt := range opts {
		opt(s)
	}
	if s.logger == nil {
		s.logger = log.DefaultLogger
	}
	client, conn, err := connect(addr, s.logger, s.dialOpts...)
	if err != nil {
		return nil, err
	}
	s.Client = client
	s.conn = conn
	return s, nil
}

func connect(addr string, logger *log.Logger, opts ...grpc.DialOption) (spherev1alpha.EnvironmentAPIClient, *grpc.ClientConn, error) {
	var sphereClient spherev1alpha.EnvironmentAPIClient
	var conn *grpc.ClientConn
	err := common.Retry(10, 1*time.Second, func() error {
		var err error
		conn, err = grpc.Dial(addr, opts...)
		if err != nil {
			return err
		}
		sphereClient = spherev1alpha.NewEnvironmentAPIClient(conn)
		resp, err := sphereClient.Info(context.Background(), &spherev1alpha.Empty{})
		if err != nil {
			return err
		}
		logger.Successf("connected to server %q", resp.ServerName)
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	return sphereClient, conn, nil
}

// Close the server, removing any local resources.
func (s *Server) Close() error {
	if s.resource != nil {
		return s.resource.Close()
	}
	if s.pool != nil {
		return s.pool.Client.RemoveContainer(dc.RemoveContainerOptions{ID: s.containerID, Force: true, RemoveVolumes: true})
	}
	return s.conn.Close()
}

// ServerOpts are the connection opts.
type ServerOpts func(*Server)

// WithServerLogger adds a logger to the server.
func WithServerLogger(logger *log.Logger) func(*Server) {
	return func(s *Server) {
		s.logger = logger
	}
}

// WithDialOpts adds dial options to the server connection.
func WithDialOpts(dialOpts ...grpc.DialOption) func(*Server) {
	return func(s *Server) {
		s.dialOpts = dialOpts
	}
}
