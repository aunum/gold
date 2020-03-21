# Env

Env provides the environment interfaces for the agents. It contains a means of spinning up local 
[Sphere](github.com/aunum/sphere) servers and an extra wrapper around the sphere gRPC interface.

The package also includes normalization tools for transorming incoming state in `norm.go`. For instance 
one could expand the dimensions of a tensor or discretize continuous state.

## Usage

Create a local Gym server in a docker container and connect to it.
```go
server, _ := NewLocalServer(GymServerConfig)
defer server.Close()
```

Find a local Gym server or create one.
```go
server, _ := FindOrCreate(GymServerConfig)
defer server.Close()
```

Create an environment
```go
env, _ := server.Make("CartPole-v0")
```

Create an environment with a normalizer which will expand the dimensions along the 0 axis.
```go
env, _ := server.Make("CartPole-v0", WithNormalizer(envv1.NewExpandDimsNormalizer(0)))
```

Create an environment and start recording video and statistics.
```go
env, _ := server.Make("CartPole-v0", WithRecorder())
```

Take a step in the env.
```go
outcome, _ := env.Step(action)
```

Reset the env.
```go
init, _ := env.Reset()
```

End the env, printing results if present and then deleting any videos and statistics.
```go
env.End()
```